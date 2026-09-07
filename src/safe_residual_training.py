from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.algorithms.common import save_checkpoint
from src.algorithms.residual_sac import transition_next_observation
from src.algorithms.safe_residual_sac import update_safe_residual_sac
from src.data.offline import NormalizationStats, PenOfflineData
from src.data.replay import FeatureReplayBuffer
from src.envs.adroit import VisualAdroitEnv
from src.evaluation import policy_inputs
from src.models.actor import SquashedGaussianActor
from src.models.critic import DoubleQCritic
from src.models.residual import ResidualGaussianActor
from src.models.safe_residual import SafeFrozenAWACResidualPolicy
from src.residual_training import (
    _assert_base_unchanged,
    _base_checkpoint_path,
    _base_snapshot,
    _capture_rng,
    _mixed_batch,
    _observation_feature,
    _restore_rng,
    _save_yaml,
    build_offline_feature_pool,
    load_frozen_awac,
)
from src.safe_evaluation import evaluate_safe_policy
from src.utils import CSVLogger, load_yaml, package_versions, resolve_device, save_json, seed_everything


def _rho_effective(step: int, warmup_steps: int, ramp_steps: int, rho: float) -> float:
    if step <= warmup_steps:
        return 0.0
    if step >= warmup_steps + ramp_steps:
        return float(rho)
    return float(rho) * (step - warmup_steps) / float(ramp_steps)


def _safe_policy(
    base_actor: SquashedGaussianActor,
    config: dict[str, Any],
    device: torch.device,
) -> SafeFrozenAWACResidualPolicy:
    feature_dim = (
        int(base_actor.visual_encoder.output_dim)
        + int(base_actor.proprio_dim)
        + int(base_actor.action_dim)
    )
    residual_actor = ResidualGaussianActor(
        input_dim=feature_dim,
        action_dim=int(base_actor.action_dim),
        **config["residual_actor"],
    ).to(device)
    return SafeFrozenAWACResidualPolicy(
        base_actor,
        residual_actor,
        rho=float(config["safe_residual"]["rho"]),
        boundary_threshold=float(config["diagnostics"]["boundary_threshold"]),
    ).to(device)


def _episode_diagnostics(official: list[bool], dropped: list[bool]) -> dict[str, float]:
    entered = any(official)
    first = next((index for index, value in enumerate(official) if value), None)
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


def _safety_decision(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    safety = config["safety_thresholds"]
    failures: list[str] = []
    if metrics["benchmark_success"] < baseline["benchmark_success"] - float(
        safety["performance_tolerance"]
    ):
        failures.append("benchmark")
    if metrics["strict_stable_success"] < baseline["strict_stable_success"] - float(
        safety["performance_tolerance"]
    ):
        failures.append("strict")
    if metrics["goal_entry_rate"] < baseline["goal_entry_rate"] - float(
        safety["goal_entry_tolerance"]
    ):
        failures.append("goal_entry")
    if metrics["correction_rms_max_dim"] > float(safety["correction_rms_max"]):
        failures.append("correction_rms")
    if metrics["z_extreme_rate"] > float(safety["z_extreme_rate_max"]):
        failures.append("z_extreme_rate")
    if metrics["numerical_clamp_count"] != 0:
        failures.append("numerical_clamp")
    if metrics["residual_added_boundary_rate"] > float(
        safety["residual_added_boundary_rate_max"]
    ):
        failures.append("residual_added_boundary")
    calibration = metrics.get("critic_calibration", {})
    if not calibration.get("pass", False):
        failures.append("critic_calibration")
    rescue_only = bool(failures) and set(failures).issubset(
        {"correction_rms", "z_extreme_rate"}
    )
    return {
        "pass": not failures,
        "failures": failures,
        "anchor_rescue_allowed": rescue_only,
        "baseline_benchmark": baseline["benchmark_success"],
        "baseline_strict": baseline["strict_stable_success"],
        "candidate_benchmark": metrics["benchmark_success"],
        "candidate_strict": metrics["strict_stable_success"],
    }


def _checkpoint_payload(
    *,
    seed: int,
    config: dict[str, Any],
    base_path: Path,
    base_checkpoint: dict[str, Any],
    policy: SafeFrozenAWACResidualPolicy,
    critic: DoubleQCritic,
    target_critic: DoubleQCritic,
    actor_optimizer,
    critic_optimizer,
    q_scale_ema: torch.Tensor,
    replay: FeatureReplayBuffer,
    rng: np.random.Generator,
    env: VisualAdroitEnv,
    online_steps: int,
    episode_index: int,
    rollout_state: dict[str, Any],
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    critic_gate_passed: bool,
    decision: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "stage": "safe_residual_sac",
        "seed": seed,
        "line": "vision",
        "config": config,
        "base_checkpoint": str(base_path),
        "base_actor_init": base_checkpoint["actor_init"],
        "normalization": base_checkpoint["normalization"],
        "residual_actor_init": policy.residual_actor.init_config(),
        "residual_actor_state": policy.residual_actor.state_dict(),
        "rho": policy.rho,
        "boundary_threshold": policy.boundary_threshold,
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
        },
        "q_scale_ema": q_scale_ema.detach().cpu(),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "decision": decision,
        "critic_gate_passed": bool(critic_gate_passed),
        "progress": {
            "online_steps": int(online_steps),
            "episode_index": int(episode_index),
        },
        "replay_state": replay.state_dict(),
        "rng_states": _capture_rng(rng),
        "environment_state": env.training_state(),
        "rollout_state": copy.deepcopy(rollout_state),
    }


def load_safe_residual_policy(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[SafeFrozenAWACResidualPolicy, NormalizationStats, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_actor, _ = load_frozen_awac(checkpoint["base_checkpoint"], device)
    residual_actor = ResidualGaussianActor(**checkpoint["residual_actor_init"]).to(device)
    residual_actor.load_state_dict(checkpoint["residual_actor_state"])
    policy = SafeFrozenAWACResidualPolicy(
        base_actor,
        residual_actor,
        rho=float(checkpoint["rho"]),
        boundary_threshold=float(checkpoint["boundary_threshold"]),
    ).to(device)
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    return policy, stats, checkpoint


def train_safe_residual_sac(
    config_path: str,
    seed: int,
    *,
    resume: bool = False,
    online_steps: int | None = None,
    lambda_anchor: float | None = None,
    run_root: str | None = None,
    smoke: bool = False,
) -> Path:
    config = load_yaml(config_path)
    config["seed"] = int(seed)
    if online_steps is not None:
        config["sac"]["online_steps"] = int(online_steps)
    if lambda_anchor is not None:
        config["safe_residual"]["lambda_anchor"] = float(lambda_anchor)
    if run_root is not None:
        config["run_root"] = str(run_root)
    total_steps = int(config["sac"]["online_steps"])
    if int(config["sac"]["utd_ratio"]) != 1:
        raise ValueError("Safe Residual V3 currently implements only utd_ratio=1")
    config["evaluation"]["steps"] = list(range(0, total_steps + 1, 10_000))
    if total_steps not in config["evaluation"]["steps"]:
        config["evaluation"]["steps"].append(total_steps)
    if smoke:
        config["run_root"] = "runs/v3/smoke_safe_residual_sac"
        config["sac"].update(
            online_steps=256,
            batch_size=64,
            critic_warmup_steps=128,
            rho_ramp_steps=64,
            checkpoint_interval=128,
            log_interval=64,
        )
        config["evaluation"]["steps"] = [0, 128, 256]
        config["validation_seeds"]["count"] = 2
        total_steps = 256
    if bool(config["safe_residual"].get("learn_temperature", False)):
        raise ValueError("Safe Residual V3 requires fixed temperature")
    std_values = {
        float(config["residual_actor"]["initial_log_std"]),
        float(config["residual_actor"]["log_std_min"]),
        float(config["residual_actor"]["log_std_max"]),
    }
    if len(std_values) != 1:
        raise ValueError("Safe Residual V3 requires a fixed residual log_std")

    seed_everything(seed, bool(config.get("deterministic_torch", True)))
    device = resolve_device(config.get("device", "cuda"))
    data_config = load_yaml(config["data_config"])
    data = PenOfflineData(
        data_config["output_path"], frame_stack=int(data_config["frame_stack"])
    )
    anchor_tag = f"{float(config['safe_residual']['lambda_anchor']):.2f}".replace(".", "p")
    run_dir = Path(config["run_root"]) / f"anchor_{anchor_tag}_seed_{seed}"
    last_path = run_dir / "last_safe_residual_sac.pt"
    if run_dir.exists() and not resume and (
        last_path.exists() or (run_dir / "validation_curve.csv").exists()
    ):
        raise FileExistsError(f"Run already exists; pass --resume: {run_dir}")
    (run_dir / "learning_curves").mkdir(parents=True, exist_ok=True)

    base_path = _base_checkpoint_path(config, seed)
    base_actor, base_checkpoint = load_frozen_awac(base_path, device)
    if base_checkpoint["normalization"] != data.stats.to_dict():
        raise ValueError("Frozen AWAC normalization differs from reconstructed dataset")
    policy = _safe_policy(base_actor, config, device)
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
    q_scale_ema = torch.ones((), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    offline_pool = build_offline_feature_pool(policy, data, device)

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
        config["online_seeds"]["per_training_seed_stride"]
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
    baseline_metrics: dict[str, Any] | None = None
    critic_gate_passed = False
    latest_metrics: dict[str, Any] = {}

    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        checkpoint_anchor = float(
            checkpoint["config"]["safe_residual"]["lambda_anchor"]
        )
        if abs(checkpoint_anchor - float(config["safe_residual"]["lambda_anchor"])) > 1e-12:
            raise ValueError(
                "Cannot change lambda_anchor while resuming a Safe Residual run"
            )
        policy.residual_actor.load_state_dict(checkpoint["residual_actor_state"])
        critic.load_state_dict(checkpoint["critic_state"])
        target_critic.load_state_dict(checkpoint["target_critic_state"])
        actor_optimizer.load_state_dict(checkpoint["optimizer_states"]["actor"])
        critic_optimizer.load_state_dict(checkpoint["optimizer_states"]["critic"])
        q_scale_ema.copy_(checkpoint["q_scale_ema"].to(device))
        replay.load_state_dict(checkpoint["replay_state"])
        start_step = int(checkpoint["progress"]["online_steps"])
        episode_index = int(checkpoint["progress"]["episode_index"])
        env.reset(seed=online_seed_start + episode_index)
        observation = env.restore_training_state(checkpoint["environment_state"])
        rollout_state = checkpoint["rollout_state"]
        baseline_metrics = checkpoint["baseline_metrics"]
        critic_gate_passed = bool(checkpoint["critic_gate_passed"])
        latest_metrics = checkpoint.get("metrics", {})
        prior_decision = checkpoint.get("decision")
        if (
            start_step >= 30_000
            and total_steps > start_step
            and not (prior_decision and prior_decision.get("pass", False))
        ):
            raise RuntimeError(
                "Cannot continue beyond a failed 30k Safe Residual gate"
            )
        _restore_rng(checkpoint["rng_states"], rng)

    _save_yaml(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_meta.json",
        {
            "project": config["project"],
            "stage": "safe_residual_sac",
            "seed": seed,
            "base_checkpoint": str(base_path),
            "environment_id": data_config["environment_id"],
            "dataset_id": data_config["dataset_id"],
            "environment_and_packages": package_versions(),
            "policy_observation": "frozen visual latent + normalized hand proprio + AWAC mean action",
            "critic_observation": "same deployable frozen feature; no privileged state",
            "online_step_definition": "actual training env.step transitions; evaluation excluded",
            "offline_actor_updates": False,
            "online_rollout_seed_start": online_seed_start,
        },
    )
    save_json(run_dir / "normalization.json", data.stats.to_dict())

    validation_seeds = range(
        int(config["validation_seeds"]["start"]),
        int(config["validation_seeds"]["start"]) + int(config["validation_seeds"]["count"]),
    )
    warmup_steps = int(config["sac"]["critic_warmup_steps"])
    ramp_steps = int(config["sac"]["rho_ramp_steps"])
    evaluation_steps = {int(step) for step in config["evaluation"]["steps"]}
    validation_logger = CSVLogger(run_dir / "validation_curve.csv")
    training_logger = CSVLogger(run_dir / "learning_curves" / "safe_residual_sac.csv")
    update_window: list[dict[str, float]] = []
    action_window: list[dict[str, float]] = []
    episode_window: list[dict[str, float]] = []

    def save_state(
        path: Path,
        step: int,
        metrics: dict[str, Any],
        decision: dict[str, Any] | None = None,
    ) -> None:
        _assert_base_unchanged(
            base_actor,
            base_parameters,
            base_checkpoint["normalization"],
            original_normalization,
        )
        replay.flush()
        save_checkpoint(
            path,
            _checkpoint_payload(
                seed=seed,
                config=config,
                base_path=base_path,
                base_checkpoint=base_checkpoint,
                policy=policy,
                critic=critic,
                target_critic=target_critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                q_scale_ema=q_scale_ema,
                replay=replay,
                rng=rng,
                env=env,
                online_steps=step,
                episode_index=episode_index,
                rollout_state=rollout_state,
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                critic_gate_passed=critic_gate_passed,
                decision=decision,
            ),
        )

    def validate(step: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
        nonlocal baseline_metrics, critic_gate_passed, latest_metrics
        include_critic = step >= warmup_steps
        metrics = evaluate_safe_policy(
            policy,
            data.stats,
            validation_seeds,
            device,
            critic=critic if include_critic else None,
            gamma=float(config["sac"]["gamma"]),
            reward_scale=float(config["sac"]["reward_scale"]),
            calibration_scale_factor=float(
                config["safety_thresholds"]["q_reasonable_scale_factor"]
            ),
        )
        if baseline_metrics is None:
            baseline_metrics = metrics
        if step == warmup_steps:
            critic_gate_passed = bool(metrics["critic_calibration"]["pass"])
        decision = None
        if step in (30_000, 100_000):
            decision = _safety_decision(metrics, baseline_metrics, config)
            save_json(run_dir / f"safety_decision_{step:06d}.json", decision)
        calibration = metrics.get("critic_calibration", {})
        validation_logger.log(
            {
                "online_steps": step,
                "benchmark_success": metrics["benchmark_success"],
                "strict_stable_success": metrics["strict_stable_success"],
                "goal_entry_rate": metrics["goal_entry_rate"],
                "goal_exit_after_entry_rate": metrics["goal_exit_after_entry_rate"],
                "consecutive_success_window_rate": metrics[
                    "consecutive_success_window_rate"
                ],
                "correction_rms": metrics["correction_rms"],
                "correction_rms_max_dim": metrics["correction_rms_max_dim"],
                "z_extreme_rate": metrics["z_extreme_rate"],
                "base_boundary_rate": metrics["base_boundary_rate"],
                "executed_action_boundary_rate": metrics[
                    "executed_action_boundary_rate"
                ],
                "residual_added_boundary_rate": metrics[
                    "residual_added_boundary_rate"
                ],
                "numerical_clamp_count": metrics["numerical_clamp_count"],
                "q_base_mean": calibration.get("q_base_mean"),
                "q_base_median": calibration.get("q_base_median"),
                "q_base_abs_p99": calibration.get("q_base_abs_p99"),
                "q_executed_mean": calibration.get("q_executed_mean"),
                "predicted_q_gain_mean": calibration.get("predicted_q_gain_mean"),
                "mc_return_mean": calibration.get("mc_return_mean"),
                "mc_return_median": calibration.get("mc_return_median"),
                "q_bias_mean": calibration.get("q_bias_mean"),
                "q_mae": calibration.get("q_mae"),
                "reasonable_return_scale": calibration.get(
                    "reasonable_discounted_return_scale"
                ),
                "q_scale_ratio": calibration.get("q_p99_to_reasonable_scale"),
                "critic_calibration_pass": calibration.get("pass"),
                "safety_pass": None if decision is None else decision["pass"],
            }
        )
        save_json(run_dir / f"validation_step_{step:06d}.json", metrics)
        latest_metrics = metrics
        policy.train()
        critic.train()
        return metrics, decision

    with torch.inference_mode():
        inputs = policy_inputs(observation, "vision", data.stats, device)
        base_action = base_actor.act(**inputs, deterministic=True)
        safe_action = policy.act(**inputs, deterministic=True)
        zero_error = float((safe_action - base_action).abs().max())
    if start_step == 0 and zero_error >= 1e-6:
        raise AssertionError(f"Safe residual step-0 mismatch: {zero_error}")
    save_json(run_dir / "zero_residual_check.json", {"max_abs_action_error": zero_error})

    if start_step == 0:
        validate(0)
        save_state(run_dir / "best_safe_residual_validation.pt", 0, baseline_metrics or {})
        save_state(last_path, 0, baseline_metrics or {})

    for online_step in range(start_step + 1, total_steps + 1):
        rho_effective = _rho_effective(
            online_step, warmup_steps, ramp_steps, policy.rho
        )
        force_zero = online_step <= warmup_steps
        inputs = policy_inputs(observation, "vision", data.stats, device)
        with torch.no_grad():
            action_tensor, action_info = policy.action_and_info(
                deterministic=False,
                rho_effective=rho_effective,
                force_zero=force_zero,
                **inputs,
            )
        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        if not np.isfinite(action).all() or np.any(np.abs(action) > 1.0 + 1e-6):
            raise FloatingPointError("Safe residual produced an invalid action")
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
        correction = action_info["correction"].squeeze(0)
        z = action_info["residual_latent"].squeeze(0)
        action_window.append(
            {
                "rho_effective": rho_effective,
                "correction_rms": float(correction.square().mean().sqrt()),
                "correction_rms_max_dim_sample": float(correction.abs().max()),
                "z_extreme_rate": float((z.abs() > 0.95).float().mean()),
                "base_boundary_rate": float(
                    action_info["base_boundary"].float().mean()
                ),
                "executed_action_boundary_rate": float(
                    action_info["executed_boundary"].float().mean()
                ),
                "residual_added_boundary_rate": float(
                    action_info["residual_added_boundary"].float().mean()
                ),
                "numerical_clamp_count": float(
                    action_info["numerical_clamp"].sum()
                ),
            }
        )
        rollout_state["raw_return"] += float(reward)
        rollout_state["length"] += 1
        rollout_state["official"].append(bool(info["official_goal"]))
        rollout_state["dropped"].append(bool(info["dropped"]))

        critic_batch = _mixed_batch(
            offline_pool,
            replay,
            int(config["sac"]["batch_size"]),
            float(config["sac"]["offline_fraction"]),
            rng,
            device,
        )
        actor_batch = replay.sample(
            int(config["sac"]["batch_size"]), rng, device
        )
        actor_update = bool(
            critic_gate_passed
            and online_step > warmup_steps
            and online_step % int(config["sac"]["policy_update_interval"]) == 0
        )
        update_metrics = update_safe_residual_sac(
            residual_actor=policy.residual_actor,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            critic_batch=critic_batch,
            actor_batch=actor_batch,
            rho_effective=rho_effective,
            gamma=float(config["sac"]["gamma"]),
            tau=float(config["sac"]["tau"]),
            reward_scale=float(config["sac"]["reward_scale"]),
            temperature=float(config["safe_residual"]["temperature"]),
            correction_budget=float(config["safe_residual"]["correction_budget"]),
            lambda_anchor=float(config["safe_residual"]["lambda_anchor"]),
            q_scale_ema=q_scale_ema,
            q_scale_decay=float(config["safe_residual"]["q_scale_ema_decay"]),
            update_actor=actor_update,
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

        if online_step % int(config["sac"]["log_interval"]) == 0:
            row: dict[str, float | int] = {"online_steps": online_step}
            for prefix, window in (
                ("update", update_window),
                ("action", action_window),
                ("episode", episode_window),
            ):
                if window:
                    for key in window[0]:
                        row[f"{prefix}_{key}"] = float(
                            np.mean([entry[key] for entry in window])
                        )
            training_logger.log(row)
            update_window.clear()
            action_window.clear()
            episode_window.clear()

        decision = None
        if online_step in evaluation_steps:
            _, decision = validate(online_step)
            if online_step == warmup_steps and not critic_gate_passed:
                save_state(last_path, online_step, latest_metrics, decision)
                env.close()
                raise RuntimeError(
                    "Critic calibration failed at warm-up; actor was not enabled"
                )
            if decision is not None and decision["pass"]:
                save_state(
                    run_dir / "best_safe_residual_validation.pt",
                    online_step,
                    latest_metrics,
                    decision,
                )
        if online_step % int(config["sac"]["checkpoint_interval"]) == 0 or online_step == total_steps:
            save_state(last_path, online_step, latest_metrics, decision)

    env.close()
    return run_dir
