from __future__ import annotations

import math

import torch
from torch import nn

from src.models.safe_residual import headroom_action


class TemporalGatedVisualResidual(nn.Module):
    """Hard-gated residual student over a frozen Vision AWAC representation."""

    def __init__(
        self,
        base_actor: nn.Module,
        *,
        hidden_size: int = 128,
        rho: float = 0.05,
        initial_gate_probability: float = 0.05,
        gate_threshold: float = 0.5,
    ):
        super().__init__()
        if getattr(base_actor, "line", None) != "vision":
            raise ValueError("Gated residual student requires a Vision AWAC base")
        if not 0.0 < initial_gate_probability < 1.0:
            raise ValueError("initial_gate_probability must be in (0, 1)")
        self.base_actor = base_actor
        self.hidden_size = int(hidden_size)
        self.rho = float(rho)
        self.gate_threshold = float(gate_threshold)
        input_dim = int(base_actor.visual_encoder.output_dim) + int(
            base_actor.proprio_dim
        ) + 2 * int(base_actor.action_dim)
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, self.hidden_size, batch_first=True)
        self.residual_head = nn.Linear(self.hidden_size, base_actor.action_dim)
        self.gate_head = nn.Linear(self.hidden_size, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(
            self.gate_head.bias,
            math.log(initial_gate_probability / (1.0 - initial_gate_probability)),
        )
        self.freeze_base()

    def freeze_base(self) -> None:
        self.base_actor.eval()
        self.base_actor.requires_grad_(False)

    @torch.no_grad()
    def frozen_current(
        self,
        *,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.base_actor.eval()
        latent = self.base_actor.visual_latent(rgb)
        actor_feature = torch.cat([latent, proprio, previous_action], dim=-1)
        base_action = self.base_actor.distribution_from_features(
            actor_feature
        ).deterministic()
        return latent.detach(), base_action.detach()

    def sequence_outputs(
        self,
        sequence: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, next_hidden = self.gru(sequence, hidden)
        z = torch.tanh(self.residual_head(encoded))
        gate_logit = self.gate_head(encoded).squeeze(-1)
        return z, gate_logit, next_hidden

    def action_and_info(
        self,
        *,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        previous_action: torch.Tensor,
        previous_base_action: torch.Tensor,
        previous_executed_action: torch.Tensor,
        temporal_history: torch.Tensor | None = None,
        gate_threshold: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latent, base_action = self.frozen_current(
            rgb=rgb,
            proprio=proprio,
            previous_action=previous_action,
        )
        temporal = torch.cat(
            [latent, proprio, previous_base_action, previous_executed_action], dim=-1
        ).unsqueeze(1)
        if temporal_history is None:
            sequence = temporal.repeat(1, 4, 1)
        else:
            if temporal_history.ndim != 3 or temporal_history.shape[1] != 4:
                raise ValueError("temporal_history must have shape (B, 4, D)")
            sequence = torch.cat([temporal_history[:, 1:], temporal], dim=1)
        z_sequence, gate_sequence, _ = self.sequence_outputs(sequence)
        z = z_sequence[:, -1]
        gate_logit = gate_sequence[:, -1]
        probability = torch.sigmoid(gate_logit)
        threshold = self.gate_threshold if gate_threshold is None else float(gate_threshold)
        gate = probability >= threshold
        residual_action, action_info = headroom_action(base_action, z, self.rho)
        action = torch.where(gate.unsqueeze(-1), residual_action, base_action)
        return action, {
            "base_action": base_action,
            "residual_latent": z,
            "gate_logit": gate_logit,
            "gate_probability": probability,
            "gate": gate,
            "temporal_history": sequence.detach(),
            "correction": action - base_action,
            "numerical_clamp": action_info["numerical_clamp"],
        }

    def train(self, mode: bool = True) -> "TemporalGatedVisualResidual":
        super().train(mode)
        self.base_actor.eval()
        return self
