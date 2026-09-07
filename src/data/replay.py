from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class FeatureReplayBuffer:
    """Disk-backed replay over frozen, deployable visual-policy features.

    Features are cached only after applying the unchanged RGB preprocessing,
    frame stack, train-split normalization, and frozen AWAC encoder. This keeps
    a 1M-transition capacity practical without changing observation semantics.
    """

    def __init__(
        self,
        directory: str | Path,
        capacity: int,
        feature_dim: int,
        action_dim: int = 24,
        resume: bool = False,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.position = 0
        self.size = 0
        self.total_insertions = 0
        mode = "r+" if resume else "w+"
        specs = {
            "feature": ((self.capacity, self.feature_dim), np.float32),
            "next_feature": ((self.capacity, self.feature_dim), np.float32),
            "action": ((self.capacity, self.action_dim), np.float32),
            "reward": ((self.capacity,), np.float32),
            "terminated": ((self.capacity,), np.uint8),
            "truncated": ((self.capacity,), np.uint8),
        }
        self.arrays: dict[str, np.memmap] = {}
        for name, (shape, dtype) in specs.items():
            path = self.directory / f"{name}.mmap"
            if resume and not path.exists():
                raise FileNotFoundError(f"Replay array is missing: {path}")
            self.arrays[name] = np.memmap(path, dtype=dtype, mode=mode, shape=shape)

    def add(
        self,
        feature: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_feature: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        index = self.position
        self.arrays["feature"][index] = np.asarray(feature, dtype=np.float32)
        self.arrays["next_feature"][index] = np.asarray(next_feature, dtype=np.float32)
        self.arrays["action"][index] = np.asarray(action, dtype=np.float32)
        self.arrays["reward"][index] = float(reward)
        self.arrays["terminated"][index] = bool(terminated)
        self.arrays["truncated"][index] = bool(truncated)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_insertions += 1

    def sample(
        self, batch_size: int, rng: np.random.Generator, device: torch.device
    ) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty online replay")
        indices = rng.integers(0, self.size, size=int(batch_size))
        return {
            name: torch.as_tensor(np.asarray(array[indices]), device=device)
            for name, array in self.arrays.items()
        }

    def state_dict(self) -> dict[str, int | str]:
        return {
            "directory": str(self.directory),
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "total_insertions": self.total_insertions,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("capacity", "feature_dim", "action_dim"):
            if int(state[key]) != int(getattr(self, key)):
                raise ValueError(f"Replay {key} mismatch")
        self.position = int(state["position"])
        self.size = int(state["size"])
        self.total_insertions = int(state["total_insertions"])

    def flush(self) -> None:
        for array in self.arrays.values():
            array.flush()


class FrozenOfflineFeaturePool:
    """Small in-memory fixed pool of frozen features and demo full actions."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("Offline feature arrays must have equal lengths")
        self.arrays = arrays
        self.size = lengths.pop()

    def sample(
        self, batch_size: int, rng: np.random.Generator, device: torch.device
    ) -> dict[str, torch.Tensor]:
        indices = rng.integers(0, self.size, size=int(batch_size))
        return {
            name: torch.as_tensor(value[indices], device=device)
            for name, value in self.arrays.items()
        }
