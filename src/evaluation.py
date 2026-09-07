from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch

from src.data.offline import NormalizationStats
from src.envs.adroit import HORIZON, VisualAdroitEnv, current_metrics, make_adroit_env


def normalize(array: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    return (np.asarray(array, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )


def policy_inputs(
    observation: np.ndarray | dict[str, np.ndarray],
    line: str,
    stats: NormalizationStats,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if line == "oracle":
        if isinstance(observation, dict):
            observation = observation["privileged"]
        actor = normalize(observation, stats.oracle_mean, stats.oracle_std)
        return {"actor": torch.as_tensor(actor, device=device).unsqueeze(0)}
    if not isinstance(observation, dict):
        raise TypeError("Vision observation must be a dictionary")
    proprio = normalize(observation["proprio"], stats.proprio_mean, stats.proprio_std)
    return {
        "rgb": torch.as_tensor(observation["rgb"], device=device).unsqueeze(0),
        "proprio": torch.as_tensor(proprio, device=device).unsqueeze(0),
        "previous_action": torch.as_tensor(
            observation["previous_action"], device=device
        ).unsqueeze(0),
    }


def privileged_tensor(
    observation: np.ndarray | dict[str, np.ndarray],
    stats: NormalizationStats,
    device: torch.device,
) -> torch.Tensor:
    value = observation["privileged"] if isinstance(observation, dict) else observation
    value = normalize(value, stats.oracle_mean, stats.oracle_std)
    return torch.as_tensor(value, device=device).unsqueeze(0)


@torch.inference_mode()
def evaluate_policy(
    actor,
    line: str,
    stats: NormalizationStats,
    seeds: Iterable[int],
    device: torch.device,
    intervention: str = "normal",
    video_dir: str | Path | None = None,
    video_episodes: int = 0,
) -> dict:
    actor.eval()
    episodes: list[dict] = []
    video_dir = Path(video_dir) if video_dir is not None else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    if line == "vision" or video_episodes:
        env = VisualAdroitEnv(intervention=intervention)
    else:
        env = make_adroit_env(render=False)

    for episode_index, seed in enumerate(seeds):
        observation, _ = env.reset(seed=int(seed))
        rewards: list[float] = []
        actions: list[np.ndarray] = []
        official: list[bool] = []
        dropped: list[bool] = []
        orientation_errors: list[float] = []
        frames: list[np.ndarray] = []
        finite = True

        for _ in range(HORIZON):
            inputs = policy_inputs(observation, line, stats, device)
            action_tensor = actor.act(**inputs, deterministic=True)
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            finite = finite and bool(np.all(np.isfinite(action)))
            if not finite:
                action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            if isinstance(env, VisualAdroitEnv):
                observation, reward, terminated, truncated, info = env.step(action)
                if episode_index < video_episodes:
                    frames.append(observation["rgb"][-1].transpose(1, 2, 0))
            else:
                observation, reward, terminated, truncated, info = env.step(action)
                info = dict(info)
                info.update(current_metrics(env))
            rewards.append(float(reward))
            actions.append(action.copy())
            official.append(bool(info["official_goal"]))
            dropped.append(bool(info["dropped"]))
            orientation_errors.append(float(info["orientation_error"]))
            if terminated or truncated:
                break

        action_array = np.asarray(actions)
        first_success = next((i + 1 for i, value in enumerate(official) if value), None)
        max_consecutive = 0
        consecutive = 0
        for reached in official:
            consecutive = consecutive + 1 if reached else 0
            max_consecutive = max(max_consecutive, consecutive)
        hold_windows = [
            all(official[start : start + 20])
            for start in range(max(0, len(official) - 19))
        ]
        exited_after_entry = bool(
            first_success is not None
            and not all(official[first_success - 1 :])
        )
        strict = bool(
            len(official) >= 20
            and all(official[-20:])
            and not any(dropped)
            and finite
        )
        episode_metrics = {
            "seed": int(seed),
            "benchmark_success": bool(any(official)),
            "strict_stable_success": strict,
            "goal_exit_after_entry": exited_after_entry,
            "has_consecutive_success_window": bool(max_consecutive >= 20),
            "strict_window_fraction": float(
                np.mean(hold_windows) if hold_windows else 0.0
            ),
            "max_consecutive_goal_steps": int(max_consecutive),
            "dropped": bool(any(dropped)),
            "final_orientation_error": float(orientation_errors[-1]),
            "time_to_success": int(first_success) if first_success is not None else None,
            "return": float(sum(rewards)),
            "action_magnitude": float(np.linalg.norm(action_array, axis=1).mean()),
            "action_smoothness": float(
                np.linalg.norm(np.diff(action_array, axis=0), axis=1).mean()
                if len(action_array) > 1
                else 0.0
            ),
            "finite": finite,
        }
        episodes.append(episode_metrics)
        if video_dir is not None and frames:
            label = "success" if strict else "failure"
            imageio.mimsave(
                video_dir / f"episode_{episode_index:03d}_seed_{seed}_{label}.mp4",
                frames,
                fps=30,
                codec="libx264",
                quality=7,
            )
    env.close()

    numeric = lambda key: np.asarray([episode[key] for episode in episodes], dtype=np.float64)
    reached_times = [
        episode["time_to_success"]
        for episode in episodes
        if episode["time_to_success"] is not None
    ]
    return {
        "benchmark_success": float(numeric("benchmark_success").mean()),
        "strict_stable_success": float(numeric("strict_stable_success").mean()),
        "goal_entry_rate": float(numeric("benchmark_success").mean()),
        "goal_exit_after_entry_rate": float(
            numeric("goal_exit_after_entry")[numeric("benchmark_success").astype(bool)].mean()
            if numeric("benchmark_success").any()
            else 0.0
        ),
        "consecutive_success_window_rate": float(numeric("has_consecutive_success_window").mean()),
        "strict_window_fraction": float(numeric("strict_window_fraction").mean()),
        "drop_rate": float(numeric("dropped").mean()),
        "final_orientation_error": float(numeric("final_orientation_error").mean()),
        "time_to_success": float(np.mean(reached_times)) if reached_times else None,
        "episode_return": float(numeric("return").mean()),
        "action_magnitude": float(numeric("action_magnitude").mean()),
        "action_smoothness": float(numeric("action_smoothness").mean()),
        "non_finite_episodes": int(sum(not episode["finite"] for episode in episodes)),
        "episodes": episodes,
    }


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return center - margin, center + margin
