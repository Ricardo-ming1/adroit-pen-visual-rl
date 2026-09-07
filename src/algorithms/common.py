from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def checkpoint_payload(
    *,
    stage: str,
    seed: int,
    line: str,
    actor,
    actor_init: dict[str, Any],
    stats: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    critic=None,
    target_critic=None,
    value=None,
    optimizer_states: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    payload = {
        "stage": stage,
        "seed": seed,
        "line": line,
        "actor_init": actor_init,
        "actor_state": actor.state_dict(),
        "normalization": stats,
        "config": config,
        "metrics": metrics or {},
        "progress": progress or {},
    }
    if critic is not None:
        payload["critic_state"] = critic.state_dict()
    if target_critic is not None:
        payload["target_critic_state"] = target_critic.state_dict()
    if value is not None:
        payload["value_state"] = value.state_dict()
    if optimizer_states is not None:
        payload["optimizer_states"] = optimizer_states
    if rng is not None:
        payload["rng_states"] = {
            "numpy": rng.bit_generator.state,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
    return payload


def restore_rng_state(checkpoint: dict[str, Any], rng: np.random.Generator) -> None:
    states = checkpoint.get("rng_states")
    if not states:
        return
    rng.bit_generator.state = states["numpy"]
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states.get("cuda"):
        torch.cuda.set_rng_state_all(states["cuda"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validation_key(metrics: dict[str, Any], offline_loss: float) -> tuple[float, ...]:
    return (
        float(metrics.get("strict_stable_success", -1.0)),
        float(metrics.get("benchmark_success", -1.0)),
        float(metrics.get("episode_return", -1e30)),
        -float(offline_loss),
    )
