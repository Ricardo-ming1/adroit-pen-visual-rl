from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import torch

from src.data.offline import NormalizationStats
from src.envs.adroit import HORIZON, VisualAdroitEnv
from src.state_distillation.oracle_observation import OracleObservationAdapter


class DistilledOraclePolicy:
    """Deployment boundary: reads RGB, proprio and previous executed action only."""

    uses_privileged_state = False

    def __init__(self, estimator, oracle_actor, oracle_stats: NormalizationStats,
                 vision_stats: NormalizationStats, device: torch.device):
        self.estimator = estimator
        self.oracle_actor = oracle_actor
        self.oracle_stats = oracle_stats
        self.vision_stats = vision_stats
        self.device = device
        self.adapter = OracleObservationAdapter().to(device)
        self.frames: deque[np.ndarray] = deque(maxlen=8)
        self.proprio: deque[np.ndarray] = deque(maxlen=8)
        self.actions: deque[np.ndarray] = deque(maxlen=8)
        self.valid: deque[float] = deque(maxlen=8)

    def reset(self) -> None:
        self.frames.clear(); self.proprio.clear(); self.actions.clear(); self.valid.clear()

    @torch.inference_mode()
    def act(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        frame = np.asarray(observation["rgb"][-1], np.uint8)
        proprio = np.asarray(observation["proprio"], np.float32)
        previous = np.asarray(observation["previous_action"], np.float32)
        if not self.frames:
            for _ in range(7):
                self.frames.append(frame.copy()); self.proprio.append(proprio.copy())
                self.actions.append(np.zeros(24, np.float32)); self.valid.append(0.0)
        self.frames.append(frame.copy()); self.proprio.append(proprio.copy())
        self.actions.append(previous.copy()); self.valid.append(1.0)
        rgb = torch.as_tensor(np.stack(self.frames), device=self.device).unsqueeze(0)
        p_raw = np.stack(self.proprio)
        p_norm = (p_raw - np.asarray(self.vision_stats.proprio_mean, np.float32)) / np.asarray(self.vision_stats.proprio_std, np.float32)
        output = self.estimator(
            rgb, torch.as_tensor(p_norm, device=self.device).unsqueeze(0),
            torch.as_tensor(np.stack(self.actions), device=self.device).unsqueeze(0),
            torch.as_tensor(np.asarray(self.valid, np.float32), device=self.device).unsqueeze(0),
        )
        current_proprio = torch.as_tensor(proprio, device=self.device).unsqueeze(0)
        raw_oracle = self.adapter(current_proprio, output["primitive"])
        mean = torch.as_tensor(self.oracle_stats.oracle_mean, device=self.device)
        std = torch.as_tensor(self.oracle_stats.oracle_std, device=self.device)
        action = self.oracle_actor.act(actor=(raw_oracle - mean) / std, deterministic=True)
        return action.squeeze(0).cpu().numpy().astype(np.float32), raw_oracle.squeeze(0).cpu().numpy()


def _aggregate(episodes: list[dict]) -> dict:
    values = lambda key: np.asarray([row[key] for row in episodes], np.float64)
    entries = values("benchmark_success").astype(bool)
    return {
        "benchmark_success": float(values("benchmark_success").mean()),
        "strict_stable_success": float(values("strict_stable_success").mean()),
        "goal_entry_rate": float(entries.mean()),
        "goal_exit_after_entry_rate": float(values("goal_exit_after_entry")[entries].mean() if entries.any() else 0.0),
        "consecutive_success_window_rate": float(values("has_consecutive_success_window").mean()),
        "max_consecutive_goal_steps": float(values("max_consecutive_goal_steps").mean()),
        "drop_rate": float(values("dropped").mean()),
        "episodes": episodes,
    }


@torch.inference_mode()
def evaluate_distilled_policy(policy: DistilledOraclePolicy, seeds: Iterable[int]) -> dict:
    policy.estimator.eval(); policy.oracle_actor.eval()
    env = VisualAdroitEnv()
    episodes: list[dict] = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=int(seed)); policy.reset()
            official: list[bool] = []; dropped: list[bool] = []; finite = True
            for _ in range(HORIZON):
                action, _ = policy.act(observation)
                finite = finite and bool(np.isfinite(action).all())
                observation, _, terminated, truncated, info = env.step(np.nan_to_num(action, nan=0.0))
                official.append(bool(info["official_goal"])); dropped.append(bool(info["dropped"]))
                if terminated or truncated: break
            first = next((i for i, x in enumerate(official) if x), None)
            streak = maximum = 0
            for reached in official:
                streak = streak + 1 if reached else 0; maximum = max(maximum, streak)
            episodes.append({
                "seed": int(seed), "benchmark_success": bool(any(official)),
                "strict_stable_success": bool(len(official) >= 20 and all(official[-20:]) and not any(dropped) and finite),
                "goal_exit_after_entry": bool(first is not None and not all(official[first:])),
                "has_consecutive_success_window": bool(maximum >= 20),
                "max_consecutive_goal_steps": int(maximum), "dropped": bool(any(dropped)), "finite": finite,
            })
    finally:
        env.close()
    return _aggregate(episodes)


def paired_summary(student: dict, baseline: dict) -> dict[str, int]:
    by_seed = {int(row["seed"]): row for row in baseline["episodes"]}
    wins = losses = ties = 0
    for row in student["episodes"]:
        base = by_seed[int(row["seed"])]
        student_score = (int(row["benchmark_success"]), int(row["strict_stable_success"]))
        base_score = (int(base["benchmark_success"]), int(base["strict_stable_success"]))
        wins += student_score > base_score; losses += student_score < base_score; ties += student_score == base_score
    return {"paired_wins": int(wins), "paired_losses": int(losses), "paired_ties": int(ties)}
