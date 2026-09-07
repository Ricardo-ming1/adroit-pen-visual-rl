from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch


Split = Literal["train", "validation", "all"]
Line = Literal["oracle", "vision"]


@dataclass
class NormalizationStats:
    oracle_mean: list[float]
    oracle_std: list[float]
    proprio_mean: list[float]
    proprio_std: list[float]

    def to_dict(self) -> dict[str, list[float]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizationStats":
        return cls(**data)


class PenOfflineData:
    """In-memory view of the small 25-episode reconstructed dataset."""

    def __init__(self, path: str | Path, frame_stack: int = 4):
        self.path = Path(path)
        self.frame_stack = frame_stack
        with h5py.File(self.path, "r") as handle:
            self.episode_ids = sorted(int(key) for key in handle["episodes"].keys())
            groups = [handle["episodes"][str(i)] for i in self.episode_ids]
            self.rgb = np.stack([group["rgb"][:] for group in groups])
            self.qpos = np.stack([group["qpos"][:] for group in groups]).astype(np.float32)
            self.qvel = np.stack([group["qvel"][:] for group in groups]).astype(np.float32)
            self.observation = np.stack(
                [group["observation"][:] for group in groups]
            ).astype(np.float32)
            self.action = np.stack([group["action"][:] for group in groups]).astype(
                np.float32
            )
            self.reward = np.stack([group["reward"][:] for group in groups]).astype(
                np.float32
            )
            self.success = np.stack([group["success"][:] for group in groups]).astype(
                bool
            )
            self.splits = [str(group.attrs["split"]) for group in groups]
            self.horizon = int(handle.attrs["horizon"])
            self.action_dim = int(handle.attrs["action_dim"])
            self.valid_transitions = int(handle.attrs["valid_transitions"])

        self._episode_to_axis = {episode_id: axis for axis, episode_id in enumerate(self.episode_ids)}
        self._indices: dict[tuple[Split, bool], np.ndarray] = {}
        for split in ("train", "validation", "all"):
            axes = [
                axis
                for axis, value in enumerate(self.splits)
                if split == "all" or value == split
            ]
            self._indices[(split, False)] = np.asarray(
                [(axis, t) for axis in axes for t in range(self.horizon)], dtype=np.int64
            )
            self._indices[(split, True)] = np.asarray(
                [(axis, t) for axis in axes for t in range(self.horizon - 1)],
                dtype=np.int64,
            )

        train_axes = [axis for axis, split in enumerate(self.splits) if split == "train"]
        oracle = self.observation[train_axes].reshape(-1, self.observation.shape[-1])
        proprio = np.concatenate(
            [self.qpos[train_axes, :, :24], self.qvel[train_axes, :, :24]], axis=-1
        ).reshape(-1, 48)
        oracle_std = np.maximum(oracle.std(axis=0), 1e-4)
        proprio_std = np.maximum(proprio.std(axis=0), 1e-4)
        self.stats = NormalizationStats(
            oracle_mean=oracle.mean(axis=0).tolist(),
            oracle_std=oracle_std.tolist(),
            proprio_mean=proprio.mean(axis=0).tolist(),
            proprio_std=proprio_std.tolist(),
        )

    def __len__(self) -> int:
        return len(self._indices[("all", True)])

    def indices(self, split: Split, transitions: bool) -> np.ndarray:
        return self._indices[(split, transitions)]

    def _frames(self, axes: np.ndarray, times: np.ndarray) -> np.ndarray:
        offsets = np.arange(-self.frame_stack + 1, 1, dtype=np.int64)
        frame_times = np.maximum(times[:, None] + offsets[None, :], 0)
        frames = self.rgb[axes[:, None], frame_times]
        return frames.transpose(0, 1, 4, 2, 3)

    def batch_from_indices(
        self,
        indices: np.ndarray,
        line: Line,
        device: torch.device,
        include_next: bool = True,
    ) -> dict[str, torch.Tensor]:
        axes, times = indices[:, 0], indices[:, 1]
        stats = self.stats
        oracle_mean = np.asarray(stats.oracle_mean, dtype=np.float32)
        oracle_std = np.asarray(stats.oracle_std, dtype=np.float32)
        proprio_mean = np.asarray(stats.proprio_mean, dtype=np.float32)
        proprio_std = np.asarray(stats.proprio_std, dtype=np.float32)

        previous = np.zeros((len(indices), self.action_dim), dtype=np.float32)
        non_initial = times > 0
        previous[non_initial] = self.action[axes[non_initial], times[non_initial] - 1]
        result: dict[str, torch.Tensor] = {
            "action": torch.as_tensor(self.action[axes, times], device=device),
            "reward": torch.as_tensor(self.reward[axes, times], device=device),
            "privileged": torch.as_tensor(
                (self.observation[axes, times] - oracle_mean) / oracle_std,
                device=device,
            ),
            "episode_axis": torch.as_tensor(axes, device=device),
            "time": torch.as_tensor(times, device=device),
        }
        if line == "oracle":
            result["actor"] = result["privileged"]
        else:
            proprio = np.concatenate(
                [self.qpos[axes, times, :24], self.qvel[axes, times, :24]], axis=-1
            )
            result.update(
                rgb=torch.as_tensor(self._frames(axes, times), device=device),
                proprio=torch.as_tensor(
                    (proprio - proprio_mean) / proprio_std, device=device
                ),
                previous_action=torch.as_tensor(previous, device=device),
            )
        if include_next:
            if np.any(times >= self.horizon - 1):
                raise ValueError("Next states requested for final source states")
            next_times = times + 1
            result["next_privileged"] = torch.as_tensor(
                (self.observation[axes, next_times] - oracle_mean) / oracle_std,
                device=device,
            )
            if line == "oracle":
                result["next_actor"] = result["next_privileged"]
            else:
                next_proprio = np.concatenate(
                    [
                        self.qpos[axes, next_times, :24],
                        self.qvel[axes, next_times, :24],
                    ],
                    axis=-1,
                )
                result.update(
                    next_rgb=torch.as_tensor(
                        self._frames(axes, next_times), device=device
                    ),
                    next_proprio=torch.as_tensor(
                        (next_proprio - proprio_mean) / proprio_std, device=device
                    ),
                    next_previous_action=result["action"],
                )
        return result

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        split: Split,
        transitions: bool,
        line: Line,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        pool = self.indices(split, transitions)
        selected = pool[rng.integers(0, len(pool), size=batch_size)]
        return self.batch_from_indices(selected, line, device, include_next=transitions)

    def sequential_batches(
        self,
        batch_size: int,
        split: Split,
        transitions: bool,
        line: Line,
        device: torch.device,
    ):
        pool = self.indices(split, transitions)
        for start in range(0, len(pool), batch_size):
            yield self.batch_from_indices(
                pool[start : start + batch_size], line, device, include_next=transitions
            )
