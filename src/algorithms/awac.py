from __future__ import annotations

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
from src.models.actor import actor_inputs
from src.models.critic import soft_update
from src.utils import CSVLogger


def _critic_target(
    actor, target_critic, batch: dict, line: str, gamma: float, reward_scale: float
) -> torch.Tensor:
    with torch.no_grad():
        next_distribution = actor.distribution(**actor_inputs(batch, line, "next_"))
        next_action, _ = next_distribution.rsample()
        next_q = target_critic.minimum(batch["next_privileged"], next_action)
        # All dataset transitions end before the source time-limit transition.
        # AdroitHandPen-v1 has no true terminations, so all 4,975 targets bootstrap.
        return reward_scale * batch["reward"] + gamma * next_q


def _critic_update(
    actor,
    critic,
    target_critic,
    optimizer,
    batch: dict,
    line: str,
    gamma: float,
    tau: float,
    reward_scale: float,
) -> dict[str, float]:
    target = _critic_target(actor, target_critic, batch, line, gamma, reward_scale)
    q1, q2 = critic(batch["privileged"], batch["action"])
    loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    optimizer.step()
    soft_update(target_critic, critic, tau)
    values = torch.cat([q1.detach(), q2.detach()])
    return {
        "critic_loss": float(loss.detach()),
        "q_mean": float(values.mean()),
        "q_std": float(values.std()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "non_finite": int(not torch.isfinite(loss).item())
        + int((~torch.isfinite(values)).sum().item())
        + int((~torch.isfinite(target)).sum().item()),
    }


def warmup_critic(
    *,
    actor,
    actor_init: dict,
    critic,
    target_critic,
    data,
    line: str,
    config: dict,
    full_config: dict,
    seed: int,
    device: torch.device,
    run_dir: Path,
    rng: np.random.Generator,
    resume: bool = False,
) -> None:
    actor.eval().requires_grad_(False)
    critic.train()
    optimizer = torch.optim.Adam(critic.parameters(), lr=float(config["critic_lr"]))
    logger = CSVLogger(run_dir / "learning_curves" / "critic_warmup.csv")
    window: list[dict[str, float]] = []
    start_update = 1
    last_path = run_dir / "last_critic_warmup.pt"
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state"])
        critic.load_state_dict(checkpoint["critic_state"])
        target_critic.load_state_dict(checkpoint["target_critic_state"])
        optimizer.load_state_dict(checkpoint["optimizer_states"]["critic"])
        restore_rng_state(checkpoint, rng)
        start_update = int(checkpoint["progress"]["update"]) + 1

    for update in range(start_update, int(config["updates"]) + 1):
        batch = data.sample(
            int(config["batch_size"]), rng, "train", True, line, device
        )
        metrics = _critic_update(
            actor,
            critic,
            target_critic,
            optimizer,
            batch,
            line,
            float(config["gamma"]),
            float(config["tau"]),
            float(config.get("reward_scale", 1.0)),
        )
        if metrics["non_finite"]:
            raise FloatingPointError(f"Non-finite critic values at update {update}: {metrics}")
        window.append(metrics)
        if update % int(config["log_interval"]) == 0 or update == int(config["updates"]):
            row = {"update": update}
            for key in window[0]:
                row[key] = float(sum(item[key] for item in window) / len(window))
            logger.log(row)
            window.clear()
        if update % 1000 == 0 or update == int(config["updates"]):
            payload = checkpoint_payload(
                stage="critic_warmup",
                seed=seed,
                line=line,
                actor=actor,
                actor_init=actor_init,
                stats=data.stats.to_dict(),
                config=full_config,
                critic=critic,
                target_critic=target_critic,
                optimizer_states={"critic": optimizer.state_dict()},
                progress={"update": update},
                rng=rng,
            )
            save_checkpoint(run_dir / "last_critic_warmup.pt", payload)
    actor.requires_grad_(True)


def train_awac(
    *,
    actor,
    actor_init: dict,
    critic,
    target_critic,
    data,
    line: str,
    config: dict,
    full_config: dict,
    seed: int,
    device: torch.device,
    run_dir: Path,
    rng: np.random.Generator,
    evaluator: Callable[[], dict],
    resume: bool = False,
) -> dict:
    actor.train()
    critic.train()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["actor_lr"]))
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["critic_lr"])
    )
    logger = CSVLogger(run_dir / "learning_curves" / "offline_awac.csv")
    best_key: tuple[float, ...] | None = None
    best_metrics: dict = {}
    window: list[dict[str, float]] = []
    start_update = 1
    last_path = run_dir / "last_offline_awac.pt"
    best_path = run_dir / "best_offline_awac_validation.pt"
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state"])
        critic.load_state_dict(checkpoint["critic_state"])
        target_critic.load_state_dict(checkpoint["target_critic_state"])
        actor_optimizer.load_state_dict(checkpoint["optimizer_states"]["actor"])
        critic_optimizer.load_state_dict(checkpoint["optimizer_states"]["critic"])
        restore_rng_state(checkpoint, rng)
        start_update = int(checkpoint["progress"]["update"]) + 1
        if best_path.exists():
            best = torch.load(best_path, map_location=device, weights_only=False)
            best_metrics = best["metrics"]
            best_key = validation_key(best_metrics, float("-inf"))

    for update in range(start_update, int(config["updates"]) + 1):
        batch = data.sample(
            int(config["batch_size"]), rng, "train", True, line, device
        )
        critic_metrics = _critic_update(
            actor,
            critic,
            target_critic,
            critic_optimizer,
            batch,
            line,
            float(config["gamma"]),
            float(config["tau"]),
            float(config.get("reward_scale", 1.0)),
        )
        if critic_metrics["non_finite"]:
            raise FloatingPointError(f"Non-finite critic values at AWAC update {update}")

        distribution = actor.distribution(**actor_inputs(batch, line))
        with torch.no_grad():
            q_data = critic.minimum(batch["privileged"], batch["action"])
            policy_q = []
            for _ in range(int(config["policy_samples"])):
                sampled_action, _ = distribution.rsample()
                policy_q.append(critic.minimum(batch["privileged"], sampled_action))
            value_pi = torch.stack(policy_q).mean(dim=0)
            advantage = q_data - value_pi
            centered = advantage - advantage.mean()
            weight = torch.exp(centered / float(config["lambda"]))
            weight = weight.clamp(max=float(config["max_advantage_weight"]))
            weight = weight / weight.mean().clamp_min(1e-6)
        if config.get("actor_objective", "weighted_log_prob") == "weighted_mse":
            per_sample_regression = (
                distribution.deterministic() - batch["action"]
            ).square().mean(dim=-1)
            actor_loss = (weight * per_sample_regression).mean()
        else:
            log_prob = distribution.log_prob(batch["action"])
            actor_loss = -(weight * log_prob).mean()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        actor_optimizer.step()

        metrics = {
            **critic_metrics,
            "actor_loss": float(actor_loss.detach()),
            "advantage_mean": float(advantage.mean()),
            "advantage_std": float(advantage.std()),
            "weight_mean": float(weight.mean()),
            "weight_max": float(weight.max()),
        }
        if not np.all(np.isfinite(list(metrics.values()))):
            raise FloatingPointError(f"Non-finite AWAC metrics at update {update}: {metrics}")
        window.append(metrics)

        should_evaluate = update % int(config["eval_interval"]) == 0 or update == int(
            config["updates"]
        )
        if should_evaluate:
            closed_loop = evaluator()
            row = {"update": update}
            for key in window[0]:
                row[key] = float(sum(item[key] for item in window) / len(window))
            row.update({key: value for key, value in closed_loop.items() if key != "episodes"})
            logger.log(row)
            window.clear()
            payload = checkpoint_payload(
                stage="offline_awac",
                seed=seed,
                line=line,
                actor=actor,
                actor_init=actor_init,
                stats=data.stats.to_dict(),
                config=full_config,
                metrics=closed_loop,
                critic=critic,
                target_critic=target_critic,
                optimizer_states={
                    "actor": actor_optimizer.state_dict(),
                    "critic": critic_optimizer.state_dict(),
                },
                progress={"update": update},
                rng=rng,
            )
            save_checkpoint(run_dir / "last_offline_awac.pt", payload)
            key = validation_key(closed_loop, actor_loss.item())
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = closed_loop
                save_checkpoint(run_dir / "best_offline_awac_validation.pt", payload)

    best = torch.load(
        run_dir / "best_offline_awac_validation.pt", map_location=device, weights_only=False
    )
    actor.load_state_dict(best["actor_state"])
    critic.load_state_dict(best["critic_state"])
    target_critic.load_state_dict(best["target_critic_state"])
    return best_metrics
