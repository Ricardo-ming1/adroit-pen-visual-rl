from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.algorithms.common import save_checkpoint
from src.algorithms.residual_sac import (
    transition_next_observation,
    update_residual_sac,
)
from src.data.offline import NormalizationStats, PenOfflineData
from src.data.replay import FeatureReplayBuffer, FrozenOfflineFeaturePool
from src.envs.adroit import HORIZON, VisualAdroitEnv
from src.evaluation import evaluate_policy, policy_inputs
from src.models.actor import SquashedGaussianActor
from src.models.critic import DoubleQCritic
from src.models.residual import FrozenAWACResidualPolicy, ResidualGaussianActor
from src.utils import CSVLogger, load_yaml, package_versions, resolve_device, save_json, seed_everything


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    temporary.replace(path)


def _base_checkpoint_path(config: dict[str, Any], seed: int) -> Path:
    run_dir = Path(str(config["base_run_template"]).format(seed=seed))
    return run_dir / str(config.get("base_checkpoint", "best_offline_awac_validation.pt"))


def load_frozen_awac(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[SquashedGaussianActor, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Frozen AWAC checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("stage") != "offline_awac" or checkpoint.get("line") != "vision":
        raise ValueError(f"Expected a vision offline-AWAC checkpoint: {checkpoint_path}")
    actor = SquashedGaussianActor(**checkpoint["actor_init"]).to(device)
    actor.load_state_dict(checkpoint["actor_state"])
    actor.eval().requires_grad_(False)
    return actor, checkpoint


def _make_policy(
    base_actor: SquashedGaussianActor,
    config: dict[str, Any],
    device: torch.device,
) -> FrozenAWACResidualPolicy:
    feature_dim = int(base_actor.visual_encoder.output_dim) + int(base_actor.proprio_dim) + int(base_actor.action_dim)
    residual_actor = ResidualGaussianActor(
        input_dim=feature_dim,
        action_dim=int(base_actor.action_dim),
        **config["residual_actor"],
    ).to(device)
    return FrozenAWACResidualPolicy(
        base_actor, residual_actor, float(config["sac"]["residual_scale"])
    ).to(device)


@torch.inference_mode()
def _observation_feature(
    policy: FrozenAWACResidualPolicy,
    observation: dict[str, np.ndarray],
    stats: NormalizationStats,
    device: torch.device,
) -> np.ndarray:
    inputs = policy_inputs(observation, "vision", stats, device)
    feature, _ = policy.frozen_feature_and_base(**inputs)
    return feature.squeeze(0).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def build_offline_feature_pool(
    policy: FrozenAWACResidualPolicy,
    data: PenOfflineData,
    device: torch.device,
    batch_size: int = 512,
) -> FrozenOfflineFeaturePool:
    collected: dict[str, list[np.ndarray]] = {
        "feature": [],
        "next_feature": [],
        "action": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
    }
    for batch in data.sequential_batches(
        batch_size, "train", True, "vision", device
    ):
        feature, _ = policy.frozen_feature_and_base(
            rgb=batch["rgb"],
            proprio=batch["proprio"],
            previous_action=batch["previous_action"],
        )
        next_feature, _ = policy.frozen_feature_and_base(
            rgb=batch["next_rgb"],
            proprio=batch["next_proprio"],
            previous_action=batch["next_previous_action"],
        )
        count = len(feature)
        collected["feature"].append(feature.cpu().numpy().astype(np.float32))
        collected["next_feature"].append(next_feature.cpu().numpy().astype(np.float32))
        # Demonstrator actions remain complete normalized environment actions.
        collected["action"].append(batch["action"].cpu().numpy().astype(np.float32))
        collected["reward"].append(batch["reward"].cpu().numpy().astype(np.float32))
        collected["terminated"].append(np.zeros(count, dtype=np.uint8))
        collected["truncated"].append(np.zeros(count, dtype=np.uint8))
    return FrozenOfflineFeaturePool(
        {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    )


def _mixed_batch(
    offline: FrozenOfflineFeaturePool,
    online: FeatureReplayBuffer,
    batch_size: int,
    offline_fraction: float,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    offline_count = int(round(batch_size * offline_fraction))
    online_count = batch_size - offline_count
    if offline_count <= 0 or online_count <= 0:
        raise ValueError("Residual SAC requires non-empty offline and online batch portions")
    first = offline.sample(offline_count, rng, device)
    second = online.sample(online_count, rng, device)
    batch = {key: torch.cat([first[key], second[key]], dim=0) for key in first}
    permutation = torch.randperm(batch_size, device=device)
    return {key: value[permutation] for key, value in batch.items()}


def _capture_rng(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "numpy_generator": rng.bit_generator.state,
        "numpy_global": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(states: dict[str, Any], rng: np.random.Generator) -> None:
    rng.bit_generator.state = states["numpy_generator"]
    np.random.set_state(states["numpy_global"])
    random.setstate(states["python"])
    torch.set_rng_state(states["torch"].cpu())
    if torch.cuda.is_available() and states.get("cuda"):
        torch.cuda.set_rng_state_all([state.cpu() for state in states["cuda"]])


def _base_snapshot(actor: SquashedGaussianActor) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}


def _assert_base_unchanged(
    actor: SquashedGaussianActor,
    snapshot: dict[str, torch.Tensor],
    normalization: dict[str, Any],
    original_normalization: dict[str, Any],
) -> None:
    if actor.training or any(parameter.requires_grad for parameter in actor.parameters()):
        raise AssertionError("Frozen AWAC base left eval/frozen mode")
    for key, value in actor.state_dict().items():
        if not torch.equal(value.detach().cpu(), snapshot[key]):
            raise AssertionError(f"Frozen AWAC tensor changed: {key}")
    if normalization != original_normalization:
        raise AssertionError("Frozen observation normalizer changed")


def _episode_diagnostics(official: list[bool], dropped: list[bool]) -> dict[str, float]:
    entered = any(official)
    first = next((i for i, value in enumerate(official) if value), None)
    exited = bool(first is not None and not all(official[first:]))
    consecutive = 0
    maximum = 0
    for value in official:
        consecutive = consecutive + 1 if value else 0
        maximum = max(maximum, consecutive)
    return {
        "goal_entry": float(entered),
        "goal_exit_after_entry": float(exited),
        "consecutive_success_window": float(maximum >= 20),
        "strict_stable_success": float(
            len(official) >= 20 and all(official[-20:]) and not any(dropped)
        ),
    }


def _checkpoint_payload(
    *,
    seed: int,
    config: dict[str, Any],
    base_checkpoint: Path,
    base_checkpoint_data: dict[str, Any],
    policy: FrozenAWACResidualPolicy,
    critic: DoubleQCritic,
    target_critic: DoubleQCritic,
    actor_optimizer,
    critic_optimizer,
    temperature_optimizer,
    log_temperature: torch.Tensor,
    online_steps: int,
    metrics: dict[str, Any],
    replay: FeatureReplayBuffer,
    rng: np.random.Generator,
    env: VisualAdroitEnv,
    episode_index: int,
    rollout_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "residual_sac",
        "seed": seed,
        "line": "vision",
        "config": config,
        "base_checkpoint": str(base_checkpoint),
        "base_actor_init": base_checkpoint_data["actor_init"],
        "normalization": base_checkpoint_data["normalization"],
        "residual_actor_init": policy.residual_actor.init_config(),
        "residual_actor_state": policy.residual_actor.state_dict(),
        "residual_scale": policy.residual_scale,
        "critic_init": {
            "state_dim": policy.residual_actor.input_dim,
            "action_dim": policy.residual_actor.action_dim,
            "hidden_dims": list(config["critic"]["hidden_dims"]),
        },
        "critic_state": critic.state_dict(),
        "target_critic_state": target_critic.state_dict(),
        "optimizer_states": {
            "actor": actor_optimizer.state_dict(),
            "critic": critic_optimizer.state_dict(),
            "temperature": temperature_optimizer.state_dict(),
        },
        "log_temperature": log_temperature.detach().cpu(),
        "metrics": metrics,
        "progress": {
            "online_steps": int(online_steps),
            "episode_index": int(episode_index),
        },
        "replay_state": replay.state_dict(),
        "rng_states": _capture_rng(rng),
        "environment_state": env.training_state(),
        "rollout_state": copy.deepcopy(rollout_state),
    }


def load_residual_policy(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[FrozenAWACResidualPolicy, NormalizationStats, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_actor, _ = load_frozen_awac(checkpoint["base_checkpoint"], device)
    residual_actor = ResidualGaussianActor(**checkpoint["residual_actor_init"]).to(device)
    residual_actor.load_state_dict(checkpoint["residual_actor_state"])
    policy = FrozenAWACResidualPolicy(
        base_actor, residual_actor, float(checkpoint["residual_scale"])
    ).to(device)
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    return policy, stats, checkpoint


def train_residual_sac(
    config_path: str,
    seed: int,
    resume: bool = False,
    smoke: bool = False,
    residual_scale: float | None = None,
    online_steps: int | None = None,
    run_root: str | None = None,
) -> Path:
    config = load_yaml(config_path)
    config["seed"] = int(seed)
    if residual_scale is not None:
        config["sac"]["residual_scale"] = float(residual_scale)
    if online_steps is not None:
        config["sac"]["online_steps"] = int(online_steps)
        config["evaluation"]["steps"] = sorted(set(config["evaluation"]["steps"] + [int(online_steps)]))
    if run_root is not None:
        config["run_root"] = run_root
    if smoke:
        config["run_root"] = "runs/v2/smoke_residual_sac"
        config["sac"].update(
            online_steps=256,
            batch_size=64,
            actor_update_start=64,
            critic_warmup_steps=64,
            checkpoint_interval=128,
        )
        config["evaluation"]["steps"] = [0, 128, 256]
        config["validation_seeds"]["count"] = 2
    seed_everything(seed, bool(config.get("deterministic_torch", True)))
    device = resolve_device(config.get("device", "cuda"))
    data_config = load_yaml(config["data_config"])
    data = PenOfflineData(
        data_config["output_path"], frame_stack=int(data_config["frame_stack"])
    )
    scale_tag = str(config["sac"]["residual_scale"]).replace(".", "p")
    run_dir = Path(config["run_root"]) / f"scale_{scale_tag}_seed_{seed}"
    last_path = run_dir / "last_residual_sac.pt"
    if run_dir.exists() and not resume and (last_path.exists() or (run_dir / "validation_curve.csv").exists()):
        raise FileExistsError(f"Run already exists; pass --resume to preserve it: {run_dir}")
    (run_dir / "learning_curves").mkdir(parents=True, exist_ok=True)
    _save_yaml(run_dir / "config.yaml", config)

    base_path = _base_checkpoint_path(config, seed)
    base_actor, base_checkpoint = load_frozen_awac(base_path, device)
    if base_checkpoint["normalization"] != data.stats.to_dict():
        raise ValueError("Dataset train-split normalization does not match the frozen AWAC checkpoint")
    policy = _make_policy(base_actor, config, device)
    base_parameters = _base_snapshot(base_actor)
    original_normalization = copy.deepcopy(base_checkpoint["normalization"])
    feature_dim = policy.residual_actor.input_dim
    critic = DoubleQCritic(
        state_dim=feature_dim,
        action_dim=data.action_dim,
        hidden_dims=list(config["critic"]["hidden_dims"]),
    ).to(device)
    target_critic = critic.target_copy().to(device)
    actor_optimizer = torch.optim.Adam(
        policy.residual_actor.parameters(), lr=float(config["sac"]["actor_lr"])
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["sac"]["critic_lr"])
    )
    log_temperature = torch.tensor(
        np.log(float(config["sac"].get("initial_temperature", 0.1))),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    temperature_optimizer = torch.optim.Adam(
        [log_temperature], lr=float(config["sac"]["temperature_lr"])
    )
    rng = np.random.default_rng(seed)
    offline_pool = build_offline_feature_pool(policy, data, device)
    if offline_pool.size != len(data.indices("train", True)):
        raise AssertionError("Offline feature pool lost transitions")

    replay_dir = run_dir / "online_replay"
    replay = FeatureReplayBuffer(
        replay_dir,
        int(config["sac"]["replay_capacity"]),
        feature_dim,
        data.action_dim,
        resume=resume and last_path.exists(),
    )
    env = VisualAdroitEnv(
        image_size=int(data_config["image_size"]),
        frame_stack=int(data_config["frame_stack"]),
        horizon=int(data_config["horizon"]),
    )
    online_seed_start = int(config["online_seeds"]["start"]) + seed * int(
        config["online_seeds"].get("per_training_seed_stride", 10_000)
    )
    episode_index = 0
    observation, _ = env.reset(seed=online_seed_start)
    rollout_state: dict[str, Any] = {
        "raw_return": 0.0,
        "length": 0,
        "official": [],
        "dropped": [],
    }
    start_step = 0
    best_key: tuple[float, float, float] | None = None
    baseline_metrics: dict[str, Any] | None = None
    best_metrics: dict[str, Any] = {}

    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        policy.residual_actor.load_state_dict(checkpoint["residual_actor_state"])
        critic.load_state_dict(checkpoint["critic_state"])
        target_critic.load_state_dict(checkpoint["target_critic_state"])
        actor_optimizer.load_state_dict(checkpoint["optimizer_states"]["actor"])
        critic_optimizer.load_state_dict(checkpoint["optimizer_states"]["critic"])
        temperature_optimizer.load_state_dict(checkpoint["optimizer_states"]["temperature"])
        log_temperature.data.copy_(checkpoint["log_temperature"].to(device))
        replay.load_state_dict(checkpoint["replay_state"])
        start_step = int(checkpoint["progress"]["online_steps"])
        episode_index = int(checkpoint["progress"]["episode_index"])
        env.reset(seed=online_seed_start + episode_index)
        observation = env.restore_training_state(checkpoint["environment_state"])
        rollout_state = checkpoint["rollout_state"]
        _restore_rng(checkpoint["rng_states"], rng)
        baseline_metrics = checkpoint.get("baseline_metrics")
        best_path = run_dir / "best_residual_sac_validation.pt"
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_metrics = best["metrics"]
            best_key = (
                float(best_metrics["benchmark_success"]),
                float(best_metrics["strict_stable_success"]),
                float(best_metrics["episode_return"]),
            )

    save_json(
        run_dir / "run_meta.json",
        {
            "project": config["project"],
            "stage": "residual_sac",
            "seed": seed,
            "base_checkpoint": str(base_path),
            "environment_id": data_config["environment_id"],
            "dataset_id": data_config["dataset_id"],
            "environment_and_packages": package_versions(),
            "actor_observation": "frozen RGB latent + normalized hand proprio + deterministic AWAC base action",
            "critic_observation": "same deployable frozen feature; no privileged simulator state",
            "online_step_definition": "sum of actual training env.step transitions; evaluation excluded",
            "offline_pool_transitions": offline_pool.size,
            "online_rollout_seed_start": online_seed_start,
        },
    )
    save_json(run_dir / "normalization.json", data.stats.to_dict())

    validation_seeds = range(
        int(config["validation_seeds"]["start"]),
        int(config["validation_seeds"]["start"]) + int(config["validation_seeds"]["count"]),
    )
    evaluation_steps = {int(value) for value in config["evaluation"]["steps"]}
    total_steps = int(config["sac"]["online_steps"])
    checkpoint_interval = int(config["sac"]["checkpoint_interval"])
    validation_logger = CSVLogger(run_dir / "validation_curve.csv")
    training_logger = CSVLogger(run_dir / "learning_curves" / "residual_sac.csv")
    update_window: list[dict[str, float]] = []
    step_window: list[dict[str, float]] = []
    episode_window: list[dict[str, float]] = []

    def save_state(path: Path, step: int, metrics: dict[str, Any]) -> None:
        _assert_base_unchanged(
            base_actor,
            base_parameters,
            base_checkpoint["normalization"],
            original_normalization,
        )
        replay.flush()
        payload = _checkpoint_payload(
            seed=seed,
            config=config,
            base_checkpoint=base_path,
            base_checkpoint_data=base_checkpoint,
            policy=policy,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temperature_optimizer=temperature_optimizer,
            log_temperature=log_temperature,
            online_steps=step,
            metrics=metrics,
            replay=replay,
            rng=rng,
            env=env,
            episode_index=episode_index,
            rollout_state=rollout_state,
        )
        payload["baseline_metrics"] = baseline_metrics
        save_checkpoint(path, payload)

    def validate(step: int) -> dict[str, Any]:
        nonlocal baseline_metrics, best_key, best_metrics
        policy.eval()
        metrics = evaluate_policy(
            policy, "vision", data.stats, validation_seeds, device
        )
        if baseline_metrics is None:
            baseline_metrics = metrics
        strict_floor = float(baseline_metrics["strict_stable_success"]) - 0.02
        eligible = float(metrics["strict_stable_success"]) + 1e-12 >= strict_floor
        key = (
            float(metrics["benchmark_success"]),
            float(metrics["strict_stable_success"]),
            float(metrics["episode_return"]),
        )
        selected = bool(eligible and (best_key is None or key > best_key))
        row = {
            "online_steps": step,
            "benchmark_success": metrics["benchmark_success"],
            "strict_stable_success": metrics["strict_stable_success"],
            "goal_entry_rate": metrics["goal_entry_rate"],
            "goal_exit_after_entry_rate": metrics["goal_exit_after_entry_rate"],
            "consecutive_success_window_rate": metrics["consecutive_success_window_rate"],
            "strict_window_fraction": metrics["strict_window_fraction"],
            "drop_rate": metrics["drop_rate"],
            "episode_return": metrics["episode_return"],
            "final_orientation_error": metrics["final_orientation_error"],
            "eligible_strict_floor": int(eligible),
            "selected_at_evaluation": int(selected),
        }
        validation_logger.log(row)
        if selected:
            best_key = key
            best_metrics = metrics
            save_state(run_dir / "best_residual_sac_validation.pt", step, metrics)
        save_json(run_dir / f"validation_step_{step:06d}.json", metrics)
        policy.train()
        return metrics

    # Exact zero-residual equivalence on a real observation.
    with torch.inference_mode():
        inputs = policy_inputs(observation, "vision", data.stats, device)
        base_action = base_actor.act(**inputs, deterministic=True)
        combined = policy.act(**inputs, deterministic=True)
        error = float((combined - base_action).abs().max())
    if start_step == 0 and error >= 1e-6:
        raise AssertionError(f"Step-0 residual policy is not AWAC-equivalent: {error}")
    save_json(run_dir / "zero_residual_check.json", {"max_abs_action_error": error})

    if start_step == 0:
        validate(0)
        save_state(run_dir / "last_residual_sac.pt", 0, baseline_metrics or {})

    for online_step in range(start_step + 1, total_steps + 1):
        policy.train()
        inputs = policy_inputs(observation, "vision", data.stats, device)
        with torch.no_grad():
            action_tensor, action_info = policy.action_and_info(
                deterministic=False, **inputs
            )
        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0 + 1e-6):
            raise FloatingPointError("Residual policy produced an invalid environment action")
        next_observation, reward, terminated, truncated, info = env.step(action)
        replay_next_observation = transition_next_observation(
            next_observation, terminated, truncated, info
        )
        feature = action_info["feature"].squeeze(0).cpu().numpy().astype(np.float32)
        next_feature = _observation_feature(
            policy, replay_next_observation, data.stats, device
        )
        replay.add(
            feature,
            action,
            reward,
            next_feature,
            bool(terminated),
            bool(truncated),
        )
        residual_action = action_info["residual_action"].squeeze(0)
        step_window.append(
            {
                "raw_reward": float(reward),
                "residual_l2_executed": float(torch.linalg.vector_norm(residual_action)),
                "scaled_residual_l2_executed": float(
                    policy.residual_scale * torch.linalg.vector_norm(residual_action)
                ),
                "combined_action_saturation_rate_executed": float(
                    action_info["clipped"].float().mean()
                ),
            }
        )
        rollout_state["raw_return"] += float(reward)
        rollout_state["length"] += 1
        rollout_state["official"].append(bool(info["official_goal"]))
        rollout_state["dropped"].append(bool(info["dropped"]))

        for _ in range(int(config["sac"]["utd_ratio"])):
            batch = _mixed_batch(
                offline_pool,
                replay,
                int(config["sac"]["batch_size"]),
                float(config["sac"]["offline_fraction"]),
                rng,
                device,
            )
            update_metrics = update_residual_sac(
                residual_actor=policy.residual_actor,
                critic=critic,
                target_critic=target_critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                temperature_optimizer=temperature_optimizer,
                log_temperature=log_temperature,
                batch=batch,
                residual_scale=policy.residual_scale,
                gamma=float(config["sac"]["gamma"]),
                tau=float(config["sac"]["tau"]),
                reward_scale=float(config["sac"]["reward_scale"]),
                target_entropy=float(config["sac"]["target_entropy"]),
                update_actor=online_step > max(
                    int(config["sac"]["actor_update_start"]),
                    int(config["sac"]["critic_warmup_steps"]),
                ),
            )
            update_window.append(update_metrics)

        observation = next_observation
        if terminated or truncated:
            diagnostics = _episode_diagnostics(
                rollout_state["official"], rollout_state["dropped"]
            )
            diagnostics.update(
                raw_episode_return=float(rollout_state["raw_return"]),
                episode_length=float(rollout_state["length"]),
            )
            episode_window.append(diagnostics)
            episode_index += 1
            observation, _ = env.reset(seed=online_seed_start + episode_index)
            rollout_state = {
                "raw_return": 0.0,
                "length": 0,
                "official": [],
                "dropped": [],
            }

        if online_step % int(config["sac"].get("log_interval", 1000)) == 0:
            row: dict[str, float | int] = {"online_steps": online_step}
            for prefix, window in (
                ("update", update_window),
                ("rollout", step_window),
                ("episode", episode_window),
            ):
                if window:
                    for key in window[0]:
                        row[f"{prefix}_{key}"] = float(
                            np.mean([entry[key] for entry in window])
                        )
            training_logger.log(row)
            update_window.clear()
            step_window.clear()
            episode_window.clear()

        if online_step in evaluation_steps:
            validate(online_step)
        if online_step % checkpoint_interval == 0 or online_step == total_steps:
            save_state(run_dir / "last_residual_sac.pt", online_step, best_metrics)

    env.close()
    save_json(
        run_dir / "best_validation.json",
        {key: value for key, value in best_metrics.items() if key != "episodes"},
    )
    return run_dir
