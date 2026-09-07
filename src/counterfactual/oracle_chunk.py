from __future__ import annotations

import torch
import torch.nn as nn


class VisualOracleChunkProposer(nn.Module):
    """Deployable 5-step headroom-latent proposal; it never sees simulator state."""
    uses_privileged_state = False

    def __init__(self, temporal_dim: int = 352, hidden_size: int = 128,
                 horizon: int = 5, action_dim: int = 24):
        super().__init__(); self.horizon=horizon; self.action_dim=action_dim
        self.gru=nn.GRU(temporal_dim,hidden_size,num_layers=1,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.SiLU(),
                                nn.Linear(hidden_size,horizon*action_dim))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        _,hidden=self.gru(temporal)
        return torch.tanh(self.head(hidden[-1])).reshape(-1,self.horizon,self.action_dim)
