from __future__ import annotations

import copy

import torch
from torch import nn

from src.models.actor import mlp


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 45, action_dim: int = 24, hidden_dims: list[int] | None = None):
        super().__init__()
        self.network = mlp(state_dim + action_dim, hidden_dims or [256, 256], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=-1)).squeeze(-1)


class DoubleQCritic(nn.Module):
    def __init__(self, state_dim: int = 45, action_dim: int = 24, hidden_dims: list[int] | None = None):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def minimum(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(state, action)
        return torch.minimum(q1, q2)

    def target_copy(self) -> "DoubleQCritic":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        return target


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int = 45, hidden_dims: list[int] | None = None):
        super().__init__()
        self.network = mlp(state_dim, hidden_dims or [256, 256], 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.lerp_(source_parameter, tau)
