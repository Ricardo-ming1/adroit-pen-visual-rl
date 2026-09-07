from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch


def split_roots_by_source_episode(
    roots: list[dict[str, Any]],
    *,
    dev_fraction: float = 0.2,
    split_seed: int = 431,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split roots without leaking any source trajectory across train/dev."""
    if not 0.0 < float(dev_fraction) < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    episodes = sorted({int(root["source_episode_seed"]) for root in roots})
    if len(episodes) < 2:
        raise ValueError("At least two source episodes are required")
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(episodes, dtype=np.int64)
    rng.shuffle(shuffled)
    dev_count = max(1, min(len(episodes) - 1, round(len(episodes) * dev_fraction)))
    dev_episodes = {int(value) for value in shuffled[:dev_count]}
    train = [root for root in roots if int(root["source_episode_seed"]) not in dev_episodes]
    dev = [root for root in roots if int(root["source_episode_seed"]) in dev_episodes]
    train_sources = {int(root["source_episode_seed"]) for root in train}
    dev_sources = {int(root["source_episode_seed"]) for root in dev}
    if train_sources & dev_sources:
        raise AssertionError("Counterfactual source-episode leakage")
    return train, dev


def save_root_dataset(
    path: str | Path,
    roots: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"roots": roots, "metadata": copy.deepcopy(metadata)},
        temporary,
    )
    temporary.replace(path)


def load_root_dataset(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["roots"], payload["metadata"]


def compact_root_row(root: dict[str, Any]) -> dict[str, Any]:
    """Return public, non-simulator metadata for CSV/result summaries."""
    return {
        "root_id": root["root_id"],
        "root_type": root["root_type"],
        "source_episode_seed": int(root["source_episode_seed"]),
        "source_episode_index": int(root["source_episode_index"]),
        "elapsed_step": int(root["elapsed_step"]),
        "remaining_steps": int(root["remaining_steps"]),
        "split": root.get("split"),
        "position_error": float(root["root_metrics"]["position_error"]),
        "orientation_similarity": float(
            root["root_metrics"]["orientation_similarity"]
        ),
        "official_goal": bool(root["root_metrics"]["official_goal"]),
        "current_goal_streak": int(root["prefix"]["current_goal_streak"]),
    }
