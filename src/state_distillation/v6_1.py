from __future__ import annotations

from collections import deque
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest


SEQUENCE_LENGTH = 8


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_metrics(seed: int, official: list[bool], dropped: list[bool], finite: bool) -> dict[str, Any]:
    first = next((i for i, value in enumerate(official) if value), None)
    streak = maximum = 0
    for reached in official:
        streak = streak + 1 if reached else 0
        maximum = max(maximum, streak)
    return {
        "seed": int(seed),
        "benchmark_success": bool(any(official)),
        "strict_stable_success": bool(
            len(official) >= 20 and all(official[-20:]) and not any(dropped) and finite
        ),
        "goal_exit_after_entry": bool(first is not None and not all(official[first:])),
        "has_consecutive_success_window": bool(maximum >= 20),
        "max_consecutive_goal_steps": int(maximum),
        "dropped": bool(any(dropped)),
        "finite": bool(finite),
    }


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(episodes)
    entries = [row for row in episodes if row["benchmark_success"]]
    return {
        "episode_count": count,
        "benchmark_success_count": sum(bool(row["benchmark_success"]) for row in episodes),
        "benchmark_success": sum(bool(row["benchmark_success"]) for row in episodes) / max(count, 1),
        "strict_success_count": sum(bool(row["strict_stable_success"]) for row in episodes),
        "strict_stable_success": sum(bool(row["strict_stable_success"]) for row in episodes) / max(count, 1),
        "goal_entry_count": len(entries),
        "goal_entry_rate": len(entries) / max(count, 1),
        "goal_exit_count": sum(bool(row["goal_exit_after_entry"]) for row in entries),
        "goal_exit_after_entry_rate": (
            sum(bool(row["goal_exit_after_entry"]) for row in entries) / len(entries) if entries else 0.0
        ),
        "twenty_step_hold_count": sum(bool(row["has_consecutive_success_window"]) for row in episodes),
        "twenty_step_hold_rate": sum(bool(row["has_consecutive_success_window"]) for row in episodes) / max(count, 1),
        "mean_max_consecutive_goal_steps": float(
            np.mean([row["max_consecutive_goal_steps"] for row in episodes]) if episodes else 0.0
        ),
        "drop_count": sum(bool(row["dropped"]) for row in episodes),
        "nonfinite_count": sum(not bool(row["finite"]) for row in episodes),
        "episodes": episodes,
    }


def paired_binary_statistics(
    candidate: list[dict[str, Any]], baseline: list[dict[str, Any]], key: str,
    *, bootstrap_replicates: int, bootstrap_seed: int,
) -> dict[str, Any]:
    base = {int(row["seed"]): bool(row[key]) for row in baseline}
    seeds = [int(row["seed"]) for row in candidate]
    if set(seeds) != set(base):
        raise ValueError("paired policies do not contain the same seed set")
    cand = np.asarray([bool(row[key]) for row in candidate], np.int8)
    ref = np.asarray([base[seed] for seed in seeds], np.int8)
    delta = cand - ref
    wins, losses = int((delta == 1).sum()), int((delta == -1).sum())
    ties = int((delta == 0).sum())
    rng = np.random.default_rng(int(bootstrap_seed))
    n = len(delta)
    means = np.empty(int(bootstrap_replicates), np.float64)
    block = 2000
    for start in range(0, len(means), block):
        stop = min(start + block, len(means))
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = delta[indices].mean(axis=1)
    discordant = wins + losses
    pvalue = float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "paired_difference": float(delta.mean()),
        "paired_bootstrap_95_ci": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "exact_mcnemar_two_sided_p": pvalue,
        "discordant_pairs": discordant,
    }


class CausalHistory:
    """Eight deployable observations aligned with the V6 estimator input."""

    def __init__(self) -> None:
        self.frames: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self.proprio: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self.actions: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self.valid: deque[float] = deque(maxlen=SEQUENCE_LENGTH)

    def reset(self, observation: dict[str, np.ndarray]) -> None:
        self.frames.clear(); self.proprio.clear(); self.actions.clear(); self.valid.clear()
        frame = np.asarray(observation["rgb"][-1], np.uint8)
        proprio = np.asarray(observation["proprio"], np.float32)
        for _ in range(SEQUENCE_LENGTH - 1):
            self.frames.append(frame.copy())
            self.proprio.append(proprio.copy())
            self.actions.append(np.zeros(24, np.float32))
            self.valid.append(0.0)
        self.append(observation)

    def append(self, observation: dict[str, np.ndarray]) -> None:
        self.frames.append(np.asarray(observation["rgb"][-1], np.uint8).copy())
        self.proprio.append(np.asarray(observation["proprio"], np.float32).copy())
        self.actions.append(np.asarray(observation["previous_action"], np.float32).copy())
        self.valid.append(1.0)

    def before_current(self) -> dict[str, np.ndarray]:
        if len(self.frames) != SEQUENCE_LENGTH:
            raise RuntimeError("causal history is incomplete")
        return {
            "frames": np.stack(tuple(self.frames)[:-1]),
            "proprio": np.stack(tuple(self.proprio)[:-1]),
            "actions": np.stack(tuple(self.actions)[:-1]),
            "valid": np.asarray(tuple(self.valid)[:-1], np.float32),
        }


def restore_policy_history(policy: Any, history: dict[str, np.ndarray]) -> None:
    policy.reset()
    for frame, proprio, action, valid in zip(
        history["frames"], history["proprio"], history["actions"], history["valid"]
    ):
        policy.frames.append(np.asarray(frame, np.uint8).copy())
        policy.proprio.append(np.asarray(proprio, np.float32).copy())
        policy.actions.append(np.asarray(action, np.float32).copy())
        policy.valid.append(float(valid))


def maximum_streak(values: Iterable[bool]) -> int:
    streak = maximum = 0
    for value in values:
        streak = streak + 1 if value else 0
        maximum = max(maximum, streak)
    return maximum
