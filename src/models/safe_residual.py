from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.residual import ResidualGaussianActor


def headroom_action(
    base_action: torch.Tensor,
    residual_latent: torch.Tensor,
    rho: float,
    numerical_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Map a residual into the directional action headroom.

    For ``rho`` in [0, 1], the unprotected result is analytically bounded. The
    final clamp protects only floating-point roundoff and is separately counted.
    """
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    positive_room = 1.0 - base_action
    negative_room = 1.0 + base_action
    directional_delta = (
        F.relu(residual_latent) * positive_room
        - F.relu(-residual_latent) * negative_room
    )
    correction = float(rho) * directional_delta
    unprotected = base_action + correction
    action = unprotected.clamp(-1.0, 1.0)
    # Count every element changed by the guard clamp. The tolerance is a
    # separate severe-violation diagnostic, not a way to hide small clamps.
    numerical_clamp = action != unprotected
    severe_bound_violation = (unprotected > 1.0 + numerical_tolerance) | (
        unprotected < -1.0 - numerical_tolerance
    )
    return action, {
        "correction": correction,
        "directional_delta": directional_delta,
        "unprotected_action": unprotected,
        "numerical_clamp": numerical_clamp,
        "severe_bound_violation": severe_bound_violation,
    }


class SafeFrozenAWACResidualPolicy(nn.Module):
    """Frozen AWAC base with an analytically bounded headroom residual."""

    def __init__(
        self,
        base_actor: nn.Module,
        residual_actor: ResidualGaussianActor,
        rho: float = 0.05,
        boundary_threshold: float = 0.95,
    ):
        super().__init__()
        if getattr(base_actor, "line", None) != "vision":
            raise ValueError("Safe residual policy requires a vision AWAC base")
        if not 0.0 <= float(rho) <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        self.base_actor = base_actor
        self.residual_actor = residual_actor
        self.rho = float(rho)
        self.boundary_threshold = float(boundary_threshold)
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
        feature = torch.cat([latent, proprio, base_action], dim=-1)
        return feature.detach(), base_action.detach()

    @staticmethod
    def base_from_feature(feature: torch.Tensor, action_dim: int = 24) -> torch.Tensor:
        return feature[..., -action_dim:]

    def combine_feature(
        self,
        feature: torch.Tensor,
        residual_latent: torch.Tensor,
        rho_effective: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base_action = self.base_from_feature(feature, self.residual_actor.action_dim)
        action, info = headroom_action(
            base_action,
            residual_latent,
            self.rho if rho_effective is None else float(rho_effective),
        )
        info["base_action"] = base_action
        base_boundary = base_action.abs() >= self.boundary_threshold
        executed_boundary = action.abs() >= self.boundary_threshold
        info["base_boundary"] = base_boundary
        info["executed_boundary"] = executed_boundary
        info["residual_added_boundary"] = executed_boundary & ~base_boundary
        return action, info

    def action_and_info(
        self,
        deterministic: bool = False,
        rho_effective: float | None = None,
        force_zero: bool = False,
        **inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature, _ = self.frozen_feature_and_base(**inputs)
        distribution = self.residual_actor.distribution(feature)
        if force_zero:
            residual_latent = torch.zeros_like(distribution.mean)
            pre_tanh = torch.zeros_like(distribution.mean)
        elif deterministic:
            residual_latent = distribution.deterministic()
            pre_tanh = distribution.mean
        else:
            residual_latent, pre_tanh = distribution.rsample()
        action, info = self.combine_feature(feature, residual_latent, rho_effective)
        info.update(
            feature=feature,
            residual_latent=residual_latent,
            pre_tanh=pre_tanh,
            residual_log_prob=distribution.log_prob(residual_latent, pre_tanh),
        )
        return action, info

    def act(self, deterministic: bool = False, **inputs: torch.Tensor) -> torch.Tensor:
        action, _ = self.action_and_info(deterministic=deterministic, **inputs)
        return action

    def train(self, mode: bool = True) -> "SafeFrozenAWACResidualPolicy":
        super().train(mode)
        self.base_actor.eval()
        return self
