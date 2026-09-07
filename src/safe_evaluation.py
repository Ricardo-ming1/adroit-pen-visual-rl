from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch

from src.data.offline import NormalizationStats
from src.envs.adroit import HORIZON, VisualAdroitEnv
from src.evaluation import policy_inputs


def _discounted_returns(rewards: list[float], gamma: float, scale: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = scale * rewards[index] + gamma * running
        returns[index] = running
    return returns


@torch.inference_mode()
def evaluate_safe_policy(
    policy,
    stats: NormalizationStats,
    seeds: Iterable[int],
    device: torch.device,
    *,
    critic=None,
    gamma: float = 0.99,
    reward_scale: float = 0.02,
    calibration_scale_factor: float = 2.0,
    rho_effective: float | None = None,
    video_dir: str | Path | None = None,
    video_episodes: int = 0,
) -> dict:
    policy.eval()
    if critic is not None:
        critic.eval()
    env = VisualAdroitEnv()
    episodes: list[dict] = []
    all_corrections: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    all_base: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    numerical_clamp_count = 0
    q_base_values: list[np.ndarray] = []
    q_executed_values: list[np.ndarray] = []
    mc_values: list[np.ndarray] = []
    scaled_rewards: list[float] = []
    video_path = Path(video_dir) if video_dir is not None else None
    if video_path is not None:
        video_path.mkdir(parents=True, exist_ok=True)

    for episode_index, seed in enumerate(seeds):
        observation, _ = env.reset(seed=int(seed))
        rewards: list[float] = []
        official: list[bool] = []
        dropped: list[bool] = []
        actions: list[np.ndarray] = []
        orientation_errors: list[float] = []
        episode_q_base: list[float] = []
        episode_q_executed: list[float] = []
        frames: list[np.ndarray] = []
        finite = True

        for _ in range(HORIZON):
            inputs = policy_inputs(observation, "vision", stats, device)
            action_tensor, info = policy.action_and_info(
                deterministic=True,
                rho_effective=rho_effective,
                **inputs,
            )
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            base = info["base_action"].squeeze(0).cpu().numpy().astype(np.float32)
            correction = info["correction"].squeeze(0).cpu().numpy().astype(np.float32)
            z = info["residual_latent"].squeeze(0).cpu().numpy().astype(np.float32)
            finite = finite and bool(np.isfinite(action).all())
            numerical_clamp_count += int(info["numerical_clamp"].sum())
            all_corrections.append(correction)
            all_z.append(z)
            all_base.append(base)
            all_actions.append(action)
            if critic is not None:
                feature = info["feature"]
                episode_q_base.append(float(critic.minimum(feature, info["base_action"])))
                episode_q_executed.append(float(critic.minimum(feature, action_tensor)))

            observation, reward, terminated, truncated, step_info = env.step(action)
            rewards.append(float(reward))
            scaled_rewards.append(float(reward_scale * reward))
            actions.append(action)
            official.append(bool(step_info["official_goal"]))
            dropped.append(bool(step_info["dropped"]))
            orientation_errors.append(float(step_info["orientation_error"]))
            if episode_index < video_episodes:
                frames.append(observation["rgb"][-1].transpose(1, 2, 0))
            if terminated or truncated:
                break

        first_success = next((i + 1 for i, value in enumerate(official) if value), None)
        first_index = None if first_success is None else first_success - 1
        maximum = 0
        consecutive = 0
        for reached in official:
            consecutive = consecutive + 1 if reached else 0
            maximum = max(maximum, consecutive)
        strict = bool(
            len(official) >= 20
            and all(official[-20:])
            and not any(dropped)
            and finite
        )
        exited = bool(first_index is not None and not all(official[first_index:]))
        action_array = np.asarray(actions)
        episodes.append(
            {
                "seed": int(seed),
                "benchmark_success": bool(any(official)),
                "strict_stable_success": strict,
                "goal_exit_after_entry": exited,
                "has_consecutive_success_window": bool(maximum >= 20),
                "dropped": bool(any(dropped)),
                "return": float(sum(rewards)),
                "final_orientation_error": float(orientation_errors[-1]),
                "time_to_success": first_success,
                "action_magnitude": float(np.linalg.norm(action_array, axis=1).mean()),
                "action_smoothness": float(
                    np.linalg.norm(np.diff(action_array, axis=0), axis=1).mean()
                    if len(action_array) > 1
                    else 0.0
                ),
                "finite": finite,
            }
        )
        if critic is not None:
            q_base_values.append(np.asarray(episode_q_base, dtype=np.float64))
            q_executed_values.append(np.asarray(episode_q_executed, dtype=np.float64))
            mc_values.append(_discounted_returns(rewards, gamma, reward_scale))
        if video_path is not None and frames:
            label = "success" if strict else "failure"
            imageio.mimsave(
                video_path / f"episode_{episode_index:03d}_seed_{seed}_{label}.mp4",
                frames,
                fps=30,
                codec="libx264",
                quality=7,
            )
    env.close()

    correction_array = np.asarray(all_corrections, dtype=np.float64)
    z_array = np.asarray(all_z, dtype=np.float64)
    base_array = np.asarray(all_base, dtype=np.float64)
    executed_array = np.asarray(all_actions, dtype=np.float64)
    correction_rms_by_dim = np.sqrt(np.mean(np.square(correction_array), axis=0))
    threshold = float(policy.boundary_threshold)
    base_boundary = np.abs(base_array) >= threshold
    executed_boundary = np.abs(executed_array) >= threshold
    numeric = lambda key: np.asarray([episode[key] for episode in episodes], dtype=np.float64)
    reached_times = [
        episode["time_to_success"]
        for episode in episodes
        if episode["time_to_success"] is not None
    ]
    benchmark = numeric("benchmark_success")
    goal_exits = numeric("goal_exit_after_entry")
    result = {
        "benchmark_success": float(benchmark.mean()),
        "strict_stable_success": float(numeric("strict_stable_success").mean()),
        "goal_entry_rate": float(benchmark.mean()),
        "goal_exit_after_entry_rate": float(
            goal_exits[benchmark.astype(bool)].mean() if benchmark.any() else 0.0
        ),
        "consecutive_success_window_rate": float(
            numeric("has_consecutive_success_window").mean()
        ),
        "drop_rate": float(numeric("dropped").mean()),
        "episode_return": float(numeric("return").mean()),
        "final_orientation_error": float(numeric("final_orientation_error").mean()),
        "time_to_success": float(np.mean(reached_times)) if reached_times else None,
        "action_magnitude": float(numeric("action_magnitude").mean()),
        "action_smoothness": float(numeric("action_smoothness").mean()),
        "non_finite_episodes": int(sum(not episode["finite"] for episode in episodes)),
        "correction_rms": float(np.sqrt(np.mean(np.square(correction_array)))),
        "correction_rms_max_dim": float(correction_rms_by_dim.max()),
        "correction_rms_by_dim": correction_rms_by_dim.tolist(),
        "z_extreme_rate": float(np.mean(np.abs(z_array) > 0.95)),
        "base_boundary_rate": float(base_boundary.mean()),
        "executed_action_boundary_rate": float(executed_boundary.mean()),
        "residual_added_boundary_rate": float(
            np.mean(executed_boundary & ~base_boundary)
        ),
        "numerical_clamp_count": int(numerical_clamp_count),
        "episodes": episodes,
    }
    if critic is not None:
        q_base = np.concatenate(q_base_values)
        q_executed = np.concatenate(q_executed_values)
        mc_return = np.concatenate(mc_values)
        reward_array = np.asarray(scaled_rewards, dtype=np.float64)
        q_bias = q_base - mc_return
        reward_p99 = float(np.quantile(np.abs(reward_array), 0.99))
        reasonable_scale = max(reward_p99 / max(1.0 - gamma, 1e-6), 1.0)
        q_p99 = float(np.quantile(np.abs(q_base), 0.99))
        finite_q = bool(
            np.isfinite(q_base).all()
            and np.isfinite(q_executed).all()
            and np.isfinite(mc_return).all()
        )
        sign_consistent = bool(
            abs(float(mc_return.mean())) < 1e-8
            or float(q_base.mean()) * float(mc_return.mean()) >= 0.0
        )
        calibration_pass = bool(
            finite_q
            and sign_consistent
            and q_p99 <= float(calibration_scale_factor) * reasonable_scale
        )
        result["critic_calibration"] = {
            "q_base_mean": float(q_base.mean()),
            "q_base_median": float(np.median(q_base)),
            "q_base_abs_p99": q_p99,
            "q_executed_mean": float(q_executed.mean()),
            "predicted_q_gain_mean": float((q_executed - q_base).mean()),
            "mc_return_mean": float(mc_return.mean()),
            "mc_return_median": float(np.median(mc_return)),
            "q_bias_mean": float(q_bias.mean()),
            "q_mae": float(np.mean(np.abs(q_bias))),
            "scaled_reward_mean": float(reward_array.mean()),
            "scaled_reward_min": float(reward_array.min()),
            "scaled_reward_max": float(reward_array.max()),
            "scaled_reward_abs_p99": reward_p99,
            "reasonable_discounted_return_scale": reasonable_scale,
            "q_p99_to_reasonable_scale": float(q_p99 / reasonable_scale),
            "finite": finite_q,
            "sign_consistent": sign_consistent,
            "pass": calibration_pass,
        }
    return result
