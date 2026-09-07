from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SEQUENCE_LENGTH = 8


@dataclass
class StateTargetStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateTargetStats":
        return cls(np.asarray(value["mean"], np.float32), np.asarray(value["std"], np.float32))


def assign_split(seed: int, dev_fraction: float = 0.1) -> str:
    bucket = int(np.random.default_rng(int(seed) ^ 0x56A6).integers(0, 10_000))
    return "dev" if bucket < int(dev_fraction * 10_000) else "train"


def assert_episode_splits(handle: h5py.File) -> None:
    seen: dict[int, str] = {}
    for key in handle["episodes"]:
        group = handle["episodes"][key]
        seed, split = int(group.attrs["seed"]), str(group.attrs["split"])
        if seed in seen and seen[seed] != split:
            raise AssertionError(f"episode seed {seed} crosses splits")
        if 30_000 <= seed <= 30_099 and split == "train":
            raise AssertionError("validation bank seed leaked into train")
        seen[seed] = split


def compute_train_target_stats(paths: Iterable[str | Path]) -> StateTargetStats:
    values: list[np.ndarray] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            assert_episode_splits(handle)
            for key in handle["episodes"]:
                group = handle["episodes"][key]
                if str(group.attrs["split"]) == "train":
                    values.append(np.asarray(group["primitive"][:, :9], np.float32))
    joined = np.concatenate(values)
    return StateTargetStats(joined.mean(0), np.maximum(joined.std(0), 1e-4))


class EpisodeSequenceDataset(Dataset):
    """Lazy causal windows. An index is always owned by exactly one episode."""

    def __init__(self, paths: Iterable[str | Path], split: str, target_stats: StateTargetStats,
                 proprio_mean: np.ndarray, proprio_std: np.ndarray):
        self.paths = [Path(path) for path in paths]
        self.split = split
        self.target_stats = target_stats
        self.proprio_mean = np.asarray(proprio_mean, np.float32)
        self.proprio_std = np.asarray(proprio_std, np.float32)
        self.index: list[tuple[int, str, int]] = []
        self._handles: dict[int, h5py.File] = {}
        for file_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as handle:
                assert_episode_splits(handle)
                for key in sorted(handle["episodes"], key=lambda x: int(x)):
                    group = handle["episodes"][key]
                    if str(group.attrs["split"]) == split:
                        eligible = np.flatnonzero(np.asarray(group["train_mask"], bool)) if "train_mask" in group else range(len(group["rgb"]))
                        self.index.extend((file_index, key, int(t)) for t in eligible)

    def __len__(self) -> int:
        return len(self.index)

    def _group(self, file_index: int, key: str):
        if file_index not in self._handles:
            self._handles[file_index] = h5py.File(self.paths[file_index], "r")
        return self._handles[file_index]["episodes"][key]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        file_index, key, t = self.index[item]
        group = self._group(file_index, key)
        unbounded = np.arange(t - SEQUENCE_LENGTH + 1, t + 1)
        times = np.maximum(unbounded, 0)
        valid = unbounded >= 0
        unique_times, inverse = np.unique(times, return_inverse=True)
        rgb = np.asarray(group["rgb"][unique_times], np.uint8)[inverse].transpose(0, 3, 1, 2)
        proprio = (np.asarray(group["proprio"][unique_times], np.float32)[inverse] - self.proprio_mean) / self.proprio_std
        previous_action = np.asarray(group["previous_action"][unique_times], np.float32)[inverse]
        return {
            "rgb": torch.from_numpy(rgb.copy()),
            "proprio": torch.from_numpy(proprio.copy()),
            "previous_action": torch.from_numpy(previous_action.copy()),
            "mask": torch.from_numpy(valid.astype(np.float32)),
            "primitive": torch.from_numpy(np.asarray(group["primitive"][t], np.float32)),
            "oracle_observation": torch.from_numpy(np.asarray(group["oracle_observation"][t], np.float32)),
            "oracle_action": torch.from_numpy(np.asarray(group["oracle_action"][t], np.float32)),
            "phase": torch.tensor(int(group["phase"][t]), dtype=torch.long),
            "exit_10": torch.tensor(float(group["exit_10"][t]), dtype=torch.float32),
            "exit_20": torch.tensor(float(group["exit_20"][t]), dtype=torch.float32),
            "episode_id": torch.tensor(int(group.attrs["seed"]), dtype=torch.long),
            "time": torch.tensor(t, dtype=torch.long),
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def episode_auxiliary_labels(official: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    official = np.asarray(official, bool)
    streak = 0
    phase = np.zeros(len(official), np.int64)
    for t, reached in enumerate(official):
        streak = streak + 1 if reached else 0
        phase[t] = 2 if streak >= 20 else (1 if reached else 0)
    exit_10 = np.zeros(len(official), np.float32)
    exit_20 = np.zeros(len(official), np.float32)
    for t in range(len(official)):
        if official[t]:
            exit_10[t] = float(not np.all(official[t:min(len(official), t + 11)]))
            exit_20[t] = float(not np.all(official[t:min(len(official), t + 21)]))
    return phase, exit_10, exit_20


def append_episode(path: str | Path, episode_id: int, *, seed: int, source: str,
                   arrays: dict[str, np.ndarray], split: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["sequence_length"] = SEQUENCE_LENGTH
        episodes = handle.require_group("episodes")
        key = str(episode_id)
        if key in episodes:
            raise ValueError(f"episode {episode_id} already exists")
        group = episodes.create_group(key)
        group.attrs["seed"] = int(seed)
        group.attrs["source"] = source
        group.attrs["split"] = split or assign_split(seed)
        for name, value in arrays.items():
            value = np.asarray(value)
            group.create_dataset(name, data=value, compression="lzf" if value.nbytes > 4096 else None)
        handle.flush()
