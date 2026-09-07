from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import nn
from torch.distributions import Normal


LOG_TWO_PI = math.log(2.0 * math.pi)


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, hidden), nn.ReLU(inplace=True)])
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class VisualEncoder(nn.Module):
    def __init__(
        self,
        frame_stack: int,
        image_size: int,
        channels: list[int],
        kernels: list[int],
        strides: list[int],
        feature_dim: int,
    ):
        super().__init__()
        convs: list[nn.Module] = []
        current = frame_stack * 3
        for output, kernel, stride in zip(channels, kernels, strides):
            convs.extend(
                [nn.Conv2d(current, output, kernel_size=kernel, stride=stride), nn.ReLU(inplace=True)]
            )
            current = output
        self.convs = nn.Sequential(*convs)
        with torch.no_grad():
            dummy = torch.zeros(1, frame_stack * 3, image_size, image_size)
            flat_dim = int(self.convs(dummy).numel())
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.output_dim = feature_dim

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("RGB input must have shape (B, frame_stack, 3, H, W)")
        rgb = rgb.float().div(255.0).sub(0.5)
        rgb = rgb.flatten(1, 2)
        return self.projection(self.convs(rgb))


class TanhNormal:
    """Tanh-squashed diagonal Gaussian with a correct change of variables."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std
        self.std = log_std.exp()
        self.base = Normal(mean, self.std)

    def rsample(self) -> tuple[torch.Tensor, torch.Tensor]:
        pre_tanh = self.base.rsample()
        return torch.tanh(pre_tanh), pre_tanh

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pre_tanh = self.base.sample()
        return torch.tanh(pre_tanh), pre_tanh

    def deterministic(self) -> torch.Tensor:
        return torch.tanh(self.mean)

    def log_prob(
        self, action: torch.Tensor, pre_tanh: torch.Tensor | None = None
    ) -> torch.Tensor:
        if pre_tanh is None:
            bounded = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            pre_tanh = torch.atanh(bounded)
        base_log_prob = self.base.log_prob(pre_tanh).sum(dim=-1)
        log_jacobian = torch.log(1.0 - action.square() + 1e-6).sum(dim=-1)
        return base_log_prob - log_jacobian

    def entropy_estimate(self) -> torch.Tensor:
        action, pre_tanh = self.rsample()
        return -self.log_prob(action, pre_tanh)


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        line: Literal["oracle", "vision"],
        action_dim: int = 24,
        oracle_dim: int = 45,
        proprio_dim: int = 48,
        previous_action_dim: int = 24,
        frame_stack: int = 4,
        image_size: int = 84,
        hidden_dims: list[int] | None = None,
        cnn_channels: list[int] | None = None,
        cnn_kernels: list[int] | None = None,
        cnn_strides: list[int] | None = None,
        feature_dim: int = 256,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        **_: Any,
    ):
        super().__init__()
        self.line = line
        self.action_dim = action_dim
        self.oracle_dim = oracle_dim
        self.proprio_dim = proprio_dim
        self.previous_action_dim = previous_action_dim
        self.frame_stack = frame_stack
        self.image_size = image_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        hidden_dims = hidden_dims or [256, 256]
        if line == "oracle":
            self.visual_encoder = None
            input_dim = oracle_dim
        elif line == "vision":
            self.visual_encoder = VisualEncoder(
                frame_stack,
                image_size,
                cnn_channels or [32, 64, 64],
                cnn_kernels or [8, 4, 3],
                cnn_strides or [4, 2, 1],
                feature_dim,
            )
            input_dim = feature_dim + proprio_dim + previous_action_dim
        else:
            raise ValueError(f"Unknown actor line {line}")
        self.trunk = mlp(input_dim, hidden_dims, hidden_dims[-1])
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.log_std_head.bias)

    @property
    def deployable_low_dim(self) -> int:
        return self.proprio_dim + self.previous_action_dim if self.line == "vision" else self.oracle_dim

    def visual_latent(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return the CNN feature used by the vision policy."""
        if self.line != "vision" or self.visual_encoder is None:
            raise ValueError("Visual latents are only defined for the vision actor")
        return self.visual_encoder(rgb)

    def distribution_from_features(self, features: torch.Tensor) -> TanhNormal:
        hidden = self.trunk(features)
        mean = self.mean_head(hidden)
        raw_log_std = self.log_std_head(hidden)
        scale = 0.5 * (self.log_std_max - self.log_std_min)
        center = 0.5 * (self.log_std_max + self.log_std_min)
        log_std = center + scale * torch.tanh(raw_log_std)
        return TanhNormal(mean, log_std)

    def distribution(
        self,
        *,
        actor: torch.Tensor | None = None,
        rgb: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
    ) -> TanhNormal:
        if self.line == "oracle":
            if actor is None or actor.shape[-1] != self.oracle_dim:
                raise ValueError("Oracle actor requires a normalized 45D observation")
            features = actor
        else:
            if rgb is None or proprio is None or previous_action is None:
                raise ValueError("Vision actor requires RGB, 48D hand proprio, and previous action")
            if proprio.shape[-1] != self.proprio_dim or previous_action.shape[-1] != self.previous_action_dim:
                raise ValueError("Vision low-dimensional inputs have incorrect dimensions")
            assert self.visual_encoder is not None
            features = torch.cat(
                [self.visual_encoder(rgb), proprio, previous_action], dim=-1
            )
        return self.distribution_from_features(features)

    def act(self, deterministic: bool = False, **inputs: torch.Tensor) -> torch.Tensor:
        distribution = self.distribution(**inputs)
        if deterministic:
            return distribution.deterministic()
        action, _ = distribution.rsample()
        return action


def actor_inputs(batch: dict[str, torch.Tensor], line: str, prefix: str = "") -> dict[str, torch.Tensor]:
    if line == "oracle":
        key = "next_actor" if prefix == "next_" else "actor"
        return {"actor": batch[key]}
    return {
        "rgb": batch[f"{prefix}rgb"],
        "proprio": batch[f"{prefix}proprio"],
        "previous_action": batch[f"{prefix}previous_action"],
    }
