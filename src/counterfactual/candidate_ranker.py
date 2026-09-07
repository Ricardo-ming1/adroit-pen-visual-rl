from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


PHASES = ("far_pre_goal_failure", "near_goal", "goal_entry_pending_exit", "stable_hold")


class CandidateHead(nn.Module):
    def __init__(self, state_dim: int, candidate_count: int, hidden: int = 128,
                 descriptor_dim: int = 72):
        super().__init__()
        self.candidate_embedding = nn.Embedding(candidate_count, 16)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(descriptor_dim + 16, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.joint = nn.Sequential(
            nn.Linear(state_dim + hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.gain_head = nn.Linear(hidden, 1)
        self.harm_head = nn.Linear(hidden, 1)
        self.progress_head = nn.Linear(hidden, 3)

    def forward(self, state: torch.Tensor, descriptor: torch.Tensor,
                candidate_id: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.candidate_embedding(candidate_id)
        candidate = self.candidate_encoder(torch.cat([descriptor, embedded], dim=-1))
        joint = self.joint(torch.cat([state, candidate], dim=-1))
        return {"gain_logit": self.gain_head(joint).squeeze(-1),
                "harm_logit": self.harm_head(joint).squeeze(-1),
                "progress": self.progress_head(joint)}


class PrivilegedCandidateRanker(nn.Module):
    """Diagnostic-only ranker; simulator state is explicit in its interface."""
    uses_privileged_state = True

    def __init__(self, privileged_dim: int, candidate_count: int, hidden: int = 128,
                 descriptor_dim: int = 72):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(privileged_dim + len(PHASES), hidden), nn.LayerNorm(hidden),
            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = CandidateHead(hidden, candidate_count, hidden, descriptor_dim)

    def forward(self, privileged: torch.Tensor, phase: torch.Tensor,
                descriptor: torch.Tensor, candidate_id: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.state_encoder(torch.cat([privileged, phase], -1)),
                         descriptor, candidate_id)


class VisualCandidateRanker(nn.Module):
    """Deployable ranker: frozen AWAC latent, proprio, and action history only."""
    uses_privileged_state = False

    def __init__(self, temporal_dim: int, candidate_count: int, hidden: int = 128,
                 descriptor_dim: int = 72):
        super().__init__()
        self.temporal_encoder = nn.GRU(temporal_dim, hidden, num_layers=1, batch_first=True)
        self.head = CandidateHead(hidden, candidate_count, hidden, descriptor_dim)

    def forward(self, temporal: torch.Tensor, descriptor: torch.Tensor,
                candidate_id: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.temporal_encoder(temporal)
        return self.head(hidden[-1], descriptor, candidate_id)


def phase_one_hot(root_type: str) -> np.ndarray:
    result = np.zeros(len(PHASES), np.float32)
    result[PHASES.index(root_type)] = 1.0
    return result


@dataclass(frozen=True)
class SelectorThreshold:
    gain: float
    harm: float


def select_candidate(prob_gain: np.ndarray, prob_harm: np.ndarray,
                     threshold: SelectorThreshold) -> int:
    """Candidate zero is index 0 and is the exact fallback."""
    eligible = np.flatnonzero((prob_gain >= threshold.gain) &
                              (prob_harm <= threshold.harm))
    eligible = eligible[eligible != 0]
    if not len(eligible):
        return 0
    score = prob_gain[eligible] - prob_harm[eligible]
    return int(eligible[int(np.argmax(score))])


def deployable_input_keys() -> tuple[str, ...]:
    return ("frozen_visual_latent_history", "proprio_history",
            "previous_base_actions", "previous_executed_actions")
