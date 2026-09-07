from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.state_distillation.dataset import EpisodeSequenceDataset, StateTargetStats


class FrozenTemporalDirectAction(nn.Module):
    """A 24D bounded action head over a frozen E2 temporal representation."""

    uses_privileged_state = False
    action_dim = 24

    def __init__(self, estimator: nn.Module, hidden_dims: Iterable[int] = (128, 128)):
        super().__init__()
        self.estimator = estimator.eval().requires_grad_(False)
        layers: list[nn.Module] = []
        current = int(estimator.gru.hidden_size)
        for width in hidden_dims:
            layers.extend((nn.Linear(current, int(width)), nn.SiLU()))
            current = int(width)
        layers.append(nn.Linear(current, self.action_dim))
        self.action_head = nn.Sequential(*layers)

    def forward(self, rgb, proprio, previous_action, mask):
        with torch.no_grad():
            hidden = self.estimator(rgb, proprio, previous_action, mask)["hidden"][-1]
        return torch.tanh(self.action_head(hidden))


class DirectActionPolicy:
    """Deployment wrapper; phase and privileged state are absent from the API."""

    uses_privileged_state = False

    def __init__(self, model: FrozenTemporalDirectAction, vision_stats, device: torch.device):
        self.model = model
        self.vision_stats = vision_stats
        self.device = device
        self.frames: deque[np.ndarray] = deque(maxlen=8)
        self.proprio: deque[np.ndarray] = deque(maxlen=8)
        self.actions: deque[np.ndarray] = deque(maxlen=8)
        self.valid: deque[float] = deque(maxlen=8)

    def reset(self):
        self.frames.clear(); self.proprio.clear(); self.actions.clear(); self.valid.clear()

    @torch.inference_mode()
    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        frame = np.asarray(observation["rgb"][-1], np.uint8)
        proprio = np.asarray(observation["proprio"], np.float32)
        previous = np.asarray(observation["previous_action"], np.float32)
        if not self.frames:
            for _ in range(7):
                self.frames.append(frame.copy()); self.proprio.append(proprio.copy())
                self.actions.append(np.zeros(24, np.float32)); self.valid.append(0.0)
        self.frames.append(frame.copy()); self.proprio.append(proprio.copy())
        self.actions.append(previous.copy()); self.valid.append(1.0)
        p = np.stack(self.proprio)
        p = (p - np.asarray(self.vision_stats.proprio_mean, np.float32)) / np.asarray(self.vision_stats.proprio_std, np.float32)
        action = self.model(
            torch.as_tensor(np.stack(self.frames), device=self.device).unsqueeze(0),
            torch.as_tensor(p, device=self.device).unsqueeze(0),
            torch.as_tensor(np.stack(self.actions), device=self.device).unsqueeze(0),
            torch.as_tensor(np.asarray(self.valid, np.float32), device=self.device).unsqueeze(0),
        )
        return action.squeeze(0).cpu().numpy().astype(np.float32)


class DirectActionSequenceDataset(EpisodeSequenceDataset):
    """D1 student-visited sequences with deploy-time inputs and training-only labels."""

    def __init__(self, paths, split, target_stats: StateTargetStats, proprio_mean, proprio_std,
                 *, near_position_error: float, near_orientation_similarity: float):
        super().__init__(paths, split, target_stats, proprio_mean, proprio_std)
        self.near_position_error = float(near_position_error)
        self.near_orientation_similarity = float(near_orientation_similarity)
        self.pre_indices: list[int] = []
        self.post_indices: list[int] = []
        for item, (file_index, key, t) in enumerate(self.index):
            group = self._group(file_index, key)
            oracle = np.asarray(group["oracle_observation"][t], np.float32)
            phase = int(group["phase"][t])
            position_error = float(np.linalg.norm(oracle[24:27] - np.asarray([0.0, -0.2, 0.25], np.float32)))
            orientation_similarity = float(np.dot(oracle[33:36], oracle[36:39]))
            post = phase > 0 or (position_error <= self.near_position_error and orientation_similarity >= self.near_orientation_similarity)
            (self.post_indices if post else self.pre_indices).append(item)
        if not self.pre_indices or not self.post_indices:
            raise ValueError("phase-balanced data requires both pre-entry and near/post-entry rows")

    def __getitem__(self, item):
        result = super().__getitem__(item)
        file_index, key, t = self.index[item]
        group = self._group(file_index, key)
        result["e2_action"] = torch.from_numpy(np.asarray(group["executed_action"][t], np.float32).copy())
        result["is_post"] = torch.tensor(float(item in self._post_set), dtype=torch.float32)
        return result

    @property
    def _post_set(self):
        if not hasattr(self, "__post_set"):
            self.__post_set = frozenset(self.post_indices)
        return self.__post_set


class ExactPhaseBalancedDataset(Dataset):
    def __init__(self, source: DirectActionSequenceDataset):
        self.source = source
        self.length = 2 * max(len(source.pre_indices), len(source.post_indices))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        pool = self.source.pre_indices if item % 2 == 0 else self.source.post_indices
        return self.source[pool[(item // 2) % len(pool)]]


def phase_balanced_loss(action, batch, *, warmup: bool, preservation_coefficient: float):
    e2 = F.smooth_l1_loss(action, batch["e2_action"], reduction="none").mean(-1)
    if warmup:
        return e2.mean()
    oracle = F.smooth_l1_loss(action, batch["oracle_action"], reduction="none").mean(-1)
    post = batch["is_post"].bool()
    per = torch.where(post, oracle + float(preservation_coefficient) * e2, e2)
    return per.mean()


def verify_d1_student_source(path: str | Path) -> dict[str, int | bool]:
    episodes = transitions = 0
    with h5py.File(path, "r") as handle:
        for group in handle["episodes"].values():
            source = str(group.attrs["source"])
            if not source.startswith("d1_student_visited_"):
                raise ValueError(f"non-E2 D1 source found: {source}")
            if "executed_action" not in group or "oracle_action" not in group:
                raise ValueError("D1 episode lacks executed-action or Oracle label")
            episodes += 1; transitions += len(group["executed_action"])
    return {"episodes": episodes, "transitions": transitions, "student_action_executed": True,
            "teacher_action_executed": False}
