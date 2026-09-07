from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from src.algorithms.common import (
    checkpoint_payload,
    restore_rng_state,
    save_checkpoint,
    validation_key,
)
from src.envs.adroit import VisualAdroitEnv, make_adroit_env
from src.evaluation import policy_inputs, privileged_tensor
from src.models.actor import actor_inputs
from src.utils import CSVLogger


def _cpu_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.squeeze(0).detach().cpu().numpy() for key, value in inputs.items()}


def _rollout_batch(
    *,
    pi_old,
    value,
    line: str,
    stats,
    steps: int,
    seed_start: int,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> tuple[dict[str, np.ndarray], dict]:
    pi_old.eval().requires_grad_(False)
    value.eval()
    env = VisualAdroitEnv() if line == "vision" else make_adroit_env(render=False)
    current_seed = seed_start
    observation, _ = env.reset(seed=current_seed)
    current_seed += 1
    storage: dict[str, list] = {
        "action": [],
        "pre_tanh": [],
        "reward": [],
        "value": [],
        "next_value": [],
        "old_logprob": [],
        "terminated": [],
        "truncated": [],
        "privileged": [],
    }
    if line == "oracle":
        storage["actor"] = []
    else:
        storage.update(rgb=[], proprio=[], previous_action=[])
    completed_returns: list[float] = []
    episode_return = 0.0

    for _ in range(steps):
        inputs = policy_inputs(observation, line, stats, device)
        privileged = privileged_tensor(observation, stats, device)
        with torch.no_grad():
            distribution = pi_old.distribution(**inputs)
            action, pre_tanh = distribution.sample()
            old_logprob = distribution.log_prob(action, pre_tanh)
            value_prediction = value(privileged)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        next_observation, reward, terminated, truncated, _ = env.step(action_np)
        with torch.no_grad():
            next_value = value(privileged_tensor(next_observation, stats, device))

        for key, array in _cpu_inputs(inputs).items():
            storage[key].append(array)
        storage["privileged"].append(privileged.squeeze(0).cpu().numpy())
        storage["action"].append(action_np)
        storage["pre_tanh"].append(pre_tanh.squeeze(0).cpu().numpy())
        storage["reward"].append(float(reward))
        storage["value"].append(float(value_prediction.item()))
        storage["next_value"].append(float(next_value.item()))
        storage["old_logprob"].append(float(old_logprob.item()))
        storage["terminated"].append(bool(terminated))
        storage["truncated"].append(bool(truncated))
        episode_return += float(reward)

        if terminated or truncated:
            completed_returns.append(episode_return)
            episode_return = 0.0
            observation, _ = env.reset(seed=current_seed)
            current_seed += 1
        else:
            observation = next_observation
    env.close()

    arrays = {key: np.asarray(value) for key, value in storage.items()}
    advantages = np.zeros(steps, dtype=np.float32)
    last_advantage = 0.0
    for index in reversed(range(steps)):
        bootstrap = 0.0 if arrays["terminated"][index] else arrays["next_value"][index]
        delta = arrays["reward"][index] + gamma * bootstrap - arrays["value"][index]
        continues = not (arrays["terminated"][index] or arrays["truncated"][index])
        last_advantage = delta + gamma * gae_lambda * float(continues) * last_advantage
        advantages[index] = last_advantage
    arrays["return"] = advantages + arrays["value"].astype(np.float32)
    arrays["advantage"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    metrics = {
        "rollout_reward_mean": float(arrays["reward"].mean()),
        "rollout_completed_episodes": len(completed_returns),
        "rollout_episode_return": float(np.mean(completed_returns))
        if completed_returns
        else None,
        "terminated_count": int(arrays["terminated"].sum()),
        "truncated_count": int(arrays["truncated"].sum()),
        "online_seed_start": seed_start,
        "online_seed_end_exclusive": current_seed,
    }
    return arrays, metrics


def _tensor_minibatch(
    rollout: dict[str, np.ndarray], indices: np.ndarray, device: torch.device, line: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs: dict[str, torch.Tensor]
    if line == "oracle":
        inputs = {"actor": torch.as_tensor(rollout["actor"][indices], device=device)}
    else:
        inputs = {
            "rgb": torch.as_tensor(rollout["rgb"][indices], device=device),
            "proprio": torch.as_tensor(rollout["proprio"][indices], device=device),
            "previous_action": torch.as_tensor(
                rollout["previous_action"][indices], device=device
            ),
        }
    fields = {
        key: torch.as_tensor(rollout[key][indices], device=device)
        for key in ("privileged", "action", "pre_tanh", "old_logprob", "advantage", "return")
    }
    return inputs, fields


def train_ppo_anchor(
    *,
    actor,
    actor_init: dict,
    reference_actor,
    value,
    data,
    line: str,
    config: dict,
    full_config: dict,
    seed: int,
    device: torch.device,
    run_dir: Path,
    rng: np.random.Generator,
    evaluator: Callable[[], dict],
    online_seed_start: int,
    resume: bool = False,
) -> dict:
    reference_actor.eval().requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["actor_lr"]))
    value_optimizer = torch.optim.Adam(value.parameters(), lr=float(config["value_lr"]))
    logger = CSVLogger(run_dir / "learning_curves" / "online_ppo.csv")
    validation_logger = CSVLogger(
        run_dir / "learning_curves" / "online_ppo_validation.csv"
    )
    best_key: tuple[float, ...] | None = None
    best_metrics: dict = {}
    online_steps = 0
    next_seed_start = online_seed_start
    start_rollout = 1
    last_path = run_dir / "last_online_ppo.pt"
    best_path = run_dir / "best_online_ppo_validation.pt"

    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state"])
        value.load_state_dict(checkpoint["value_state"])
        reference_actor.load_state_dict(checkpoint["reference_actor_state"])
        actor_optimizer.load_state_dict(checkpoint["optimizer_states"]["actor"])
        value_optimizer.load_state_dict(checkpoint["optimizer_states"]["value"])
        restore_rng_state(checkpoint, rng)
        start_rollout = int(checkpoint["progress"]["rollout"]) + 1
        online_steps = int(checkpoint["progress"]["online_steps"])
        next_seed_start = int(checkpoint["progress"]["next_online_seed"])
        best = torch.load(best_path, map_location=device, weights_only=False)
        best_metrics = best["metrics"]
        best_key = validation_key(best_metrics, -best_metrics["episode_return"])
    else:
        # Include the exact offline handoff (0 online steps) in validation-based
        # model selection, so online degradation cannot silently replace pi_ref.
        initial_metrics = evaluator()
        initial_payload = checkpoint_payload(
            stage="online_ppo_anchor",
            seed=seed,
            line=line,
            actor=actor,
            actor_init=actor_init,
            stats=data.stats.to_dict(),
            config=full_config,
            metrics=initial_metrics,
            value=value,
            progress={"rollout": 0, "online_steps": 0, "next_online_seed": next_seed_start},
            rng=rng,
        )
        initial_payload["reference_actor_state"] = reference_actor.state_dict()
        save_checkpoint(best_path, initial_payload)
        best_key = validation_key(initial_metrics, -initial_metrics["episode_return"])
        best_metrics = initial_metrics
        validation_logger.log(
            {
                "rollout": 0,
                "online_steps": 0,
                **{key: value for key, value in initial_metrics.items() if key != "episodes"},
            }
        )

    for rollout_index in range(start_rollout, int(config["num_rollouts"]) + 1):
        # pi_old is an explicit frozen snapshot associated with this rollout.
        pi_old = copy.deepcopy(actor).to(device).eval().requires_grad_(False)
        rollout, rollout_metrics = _rollout_batch(
            pi_old=pi_old,
            value=value,
            line=line,
            stats=data.stats,
            steps=int(config["rollout_steps"]),
            seed_start=next_seed_start,
            device=device,
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
        )
        next_seed_start = int(rollout_metrics["online_seed_end_exclusive"])
        online_steps += int(config["rollout_steps"])
        batch_indices = np.arange(len(rollout["reward"]))
        update_metrics: list[dict[str, float]] = []
        stopped_early = False

        actor.train()
        value.train()
        for epoch in range(int(config["ppo_epochs"])):
            rng.shuffle(batch_indices)
            for start in range(0, len(batch_indices), int(config["minibatch_size"])):
                indices = batch_indices[start : start + int(config["minibatch_size"])]
                inputs, fields = _tensor_minibatch(rollout, indices, device, line)
                distribution = actor.distribution(**inputs)
                new_logprob = distribution.log_prob(
                    fields["action"], fields["pre_tanh"]
                )
                log_ratio = new_logprob - fields["old_logprob"]
                ratio = log_ratio.exp()
                # The tanh transform is bijective, so the policy KL is exactly
                # the KL between the underlying diagonal Gaussians. Check it
                # before every optimizer step rather than after a full epoch.
                with torch.no_grad():
                    old_distribution = pi_old.distribution(**inputs)
                    old_policy_kl = torch.distributions.kl_divergence(
                        old_distribution.base, distribution.base
                    ).sum(dim=-1).mean()
                if float(old_policy_kl) > 1.5 * float(
                    config["target_old_policy_kl"]
                ):
                    stopped_early = True
                    break
                unclipped = ratio * fields["advantage"]
                clipped = ratio.clamp(
                    1.0 - float(config["clip_ratio"]),
                    1.0 + float(config["clip_ratio"]),
                ) * fields["advantage"]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                entropy = distribution.entropy_estimate().mean()

                reference_distribution = reference_actor.distribution(**inputs)
                reference_kl = torch.distributions.kl_divergence(
                    distribution.base, reference_distribution.base
                ).sum(dim=-1).mean()

                if float(config["bc_aux_coef"]) > 0:
                    demonstration = data.sample(
                        len(indices), rng, "train", False, line, device
                    )
                    demonstration_distribution = actor.distribution(
                        **actor_inputs(demonstration, line)
                    )
                    bc_loss = F.mse_loss(
                        demonstration_distribution.deterministic(),
                        demonstration["action"],
                    )
                else:
                    bc_loss = torch.zeros((), device=device)

                actor_loss = (
                    policy_loss
                    - float(config["entropy_coef"]) * entropy
                    + float(config["reference_kl_coef"]) * reference_kl
                    + float(config["bc_aux_coef"]) * bc_loss
                )
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), float(config["max_grad_norm"])
                )
                actor_optimizer.step()

                predicted_value = value(fields["privileged"])
                value_loss = F.mse_loss(predicted_value, fields["return"])
                value_optimizer.zero_grad(set_to_none=True)
                (float(config["value_coef"]) * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    value.parameters(), float(config["max_grad_norm"])
                )
                value_optimizer.step()

                with torch.no_grad():
                    clip_fraction = float(
                        ((ratio - 1.0).abs() > float(config["clip_ratio"])).float().mean()
                    )
                metrics = {
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "reference_kl": float(reference_kl.detach()),
                    "bc_aux_loss": float(bc_loss.detach()),
                    "old_policy_kl": float(old_policy_kl),
                    "clip_fraction": clip_fraction,
                }
                if not np.all(np.isfinite(list(metrics.values()))):
                    raise FloatingPointError(f"Non-finite PPO update metrics: {metrics}")
                update_metrics.append(metrics)
            if stopped_early:
                break

        closed_loop = evaluator()
        row = {
            "rollout": rollout_index,
            "online_steps": online_steps,
            **rollout_metrics,
            "early_stop_old_kl": stopped_early,
        }
        for key in update_metrics[0]:
            row[key] = float(np.mean([item[key] for item in update_metrics]))
        row.update({key: value for key, value in closed_loop.items() if key != "episodes"})
        logger.log(row)
        validation_logger.log(
            {
                "rollout": rollout_index,
                "online_steps": online_steps,
                **{key: value for key, value in closed_loop.items() if key != "episodes"},
            }
        )
        payload = checkpoint_payload(
            stage="online_ppo_anchor",
            seed=seed,
            line=line,
            actor=actor,
            actor_init=actor_init,
            stats=data.stats.to_dict(),
            config=full_config,
            metrics=closed_loop,
            value=value,
            optimizer_states={
                "actor": actor_optimizer.state_dict(),
                "value": value_optimizer.state_dict(),
            },
            progress={
                "rollout": rollout_index,
                "online_steps": online_steps,
                "next_online_seed": next_seed_start,
            },
            rng=rng,
        )
        payload["reference_actor_state"] = reference_actor.state_dict()
        save_checkpoint(run_dir / "last_online_ppo.pt", payload)
        key = validation_key(closed_loop, -closed_loop["episode_return"])
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = closed_loop
            save_checkpoint(run_dir / "best_online_ppo_validation.pt", payload)

    best = torch.load(
        run_dir / "best_online_ppo_validation.pt", map_location=device, weights_only=False
    )
    actor.load_state_dict(best["actor_state"])
    value.load_state_dict(best["value_state"])
    return best_metrics
