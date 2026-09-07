from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.actor import TanhNormal, mlp


class ResidualGaussianActor(nn.Module):
    """Small squashed-Gaussian residual head over frozen AWAC features."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 24,
        hidden_dims: list[int] | None = None,
        initial_log_std: float = -3.0,
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = list(hidden_dims)
        self.initial_log_std = float(initial_log_std)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.trunk = mlp(self.input_dim, hidden_dims, hidden_dims[-1])
        self.mean_head = nn.Linear(hidden_dims[-1], self.action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], self.action_dim)
        # The deterministic residual is exactly zero at step 0.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, self.initial_log_std)

    def distribution(self, feature: torch.Tensor) -> TanhNormal:
        if feature.shape[-1] != self.input_dim:
            raise ValueError(
                f"Residual feature has dimension {feature.shape[-1]}, expected {self.input_dim}"
            )
        hidden = self.trunk(feature)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.log_std_min, self.log_std_max)
        return TanhNormal(mean, log_std)

    def act(self, feature: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        distribution = self.distribution(feature)
        if deterministic:
            return distribution.deterministic()
        action, _ = distribution.rsample()
        return action

    def init_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "initial_log_std": self.initial_log_std,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
        }


class FrozenAWACResidualPolicy(nn.Module):
    """Frozen deterministic AWAC base plus a trainable residual actor.

    Feature is ``[visual_latent, normalized_proprio, deterministic_base_action]``.
    No simulator-only observation enters either the policy or online critic.
    """

    def __init__(
        self,
        base_actor: nn.Module,
        residual_actor: ResidualGaussianActor,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        if getattr(base_actor, "line", None) != "vision":
            raise ValueError("Residual policy requires a vision AWAC base")
        self.base_actor = base_actor
        self.residual_actor = residual_actor
        self.residual_scale = float(residual_scale)
        self.freeze_base()

    def freeze_base(self) -> None:
        self.base_actor.eval()
        self.base_actor.requires_grad_(False)

    @torch.no_grad()
    def frozen_feature_and_base(
        self,
        *,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.base_actor.eval()
        latent = self.base_actor.visual_latent(rgb)
        base_features = torch.cat([latent, proprio, previous_action], dim=-1)
        base_action = self.base_actor.distribution_from_features(
            base_features
        ).deterministic()
        residual_feature = torch.cat([latent, proprio, base_action], dim=-1)
        return residual_feature.detach(), base_action.detach()

    @staticmethod
    def base_from_feature(feature: torch.Tensor, action_dim: int = 24) -> torch.Tensor:
        return feature[..., -action_dim:]

    def combine(
        self, base_action: torch.Tensor, residual_action: torch.Tensor
    ) -> torch.Tensor:
        return torch.clamp(
            base_action + self.residual_scale * residual_action, -1.0, 1.0
        )

    def action_and_info(
        self, deterministic: bool = False, **inputs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature, base_action = self.frozen_feature_and_base(**inputs)
        distribution = self.residual_actor.distribution(feature)
        if deterministic:
            residual_action = distribution.deterministic()
            pre_tanh = distribution.mean
        else:
            residual_action, pre_tanh = distribution.rsample()
        unclipped = base_action + self.residual_scale * residual_action
        action = unclipped.clamp(-1.0, 1.0)
        return action, {
            "feature": feature,
            "base_action": base_action,
            "residual_action": residual_action,
            "pre_tanh": pre_tanh,
            "residual_log_prob": distribution.log_prob(
                residual_action, pre_tanh
            ),
            "clipped": unclipped.abs() > 1.0,
        }

    def act(self, deterministic: bool = False, **inputs: torch.Tensor) -> torch.Tensor:
        action, _ = self.action_and_info(deterministic=deterministic, **inputs)
        return action

    def train(self, mode: bool = True) -> "FrozenAWACResidualPolicy":
        super().train(mode)
        # nn.Module.train() recurses into the base; restore the frozen-mode invariant.
        self.base_actor.eval()
        return self
