from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.models.actor import VisualEncoder
from src.state_distillation.oracle_observation import PRIMITIVE_DIM, rotation_6d_to_matrix


class TemporalStateEstimator(nn.Module):
    """Eight-step causal RGB/proprio/executed-action state estimator."""

    uses_privileged_state = False
    sequence_length = 8

    def __init__(self, vision_encoder: VisualEncoder, hidden_size: int = 128):
        super().__init__()
        self.visual_encoder = self._single_frame_copy(vision_encoder)
        feature_dim = int(self.visual_encoder.output_dim)
        self.gru = nn.GRU(feature_dim + 48 + 24 + 1, hidden_size, batch_first=True)
        self.state_head = nn.Sequential(nn.Linear(hidden_size, 128), nn.SiLU(), nn.Linear(128, PRIMITIVE_DIM))
        self.phase_head = nn.Linear(hidden_size, 3)
        self.exit_head = nn.Linear(hidden_size, 2)

    @staticmethod
    def _single_frame_copy(source: VisualEncoder) -> VisualEncoder:
        conv_layers = [layer for layer in source.convs if isinstance(layer, nn.Conv2d)]
        channels = [layer.out_channels for layer in conv_layers]
        kernels = [layer.kernel_size[0] for layer in conv_layers]
        strides = [layer.stride[0] for layer in conv_layers]
        # Projection dimensions are recovered from the source, while input changes
        # from a 4-frame stack to one frame at each temporal GRU step.
        image_size = 84
        target = VisualEncoder(1, image_size, channels, kernels, strides, source.output_dim)
        first_source = conv_layers[0]
        first_target = next(layer for layer in target.convs if isinstance(layer, nn.Conv2d))
        with torch.no_grad():
            stacks = first_source.weight.shape[1] // 3
            first_target.weight.copy_(first_source.weight.reshape(first_source.out_channels, stacks, 3, *first_source.kernel_size).sum(1))
            first_target.bias.copy_(first_source.bias)
        # All later conv and projection tensors are byte-compatible.
        source_state = source.state_dict()
        target_state = target.state_dict()
        for key in target_state:
            if key != "convs.0.weight" and key in source_state and source_state[key].shape == target_state[key].shape:
                target_state[key].copy_(source_state[key])
        target.load_state_dict(target_state)
        return target

    def forward(
        self,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        previous_action: torch.Tensor,
        mask: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if rgb.ndim != 5 or rgb.shape[1] != self.sequence_length or rgb.shape[2] != 3:
            raise ValueError("rgb must be (B,8,3,H,W)")
        if proprio.shape[-2:] != (self.sequence_length, 48):
            raise ValueError("proprio must be (B,8,48)")
        if previous_action.shape[-2:] != (self.sequence_length, 24):
            raise ValueError("previous_action must be (B,8,24)")
        if mask.shape[-1] != self.sequence_length:
            raise ValueError("mask must be (B,8)")
        b, t = rgb.shape[:2]
        visual = self.visual_encoder(rgb.reshape(b * t, 1, 3, rgb.shape[-2], rgb.shape[-1])).reshape(b, t, -1)
        features = torch.cat((visual, proprio, previous_action, mask.unsqueeze(-1)), dim=-1)
        encoded, hidden_out = self.gru(features, hidden)
        final = encoded[:, -1]
        primitive = self.state_head(final)
        # Always project both predicted rotations onto SO(3) before they reach the adapter.
        primitive = torch.cat((primitive[..., :9], self._canonical_6d(primitive[..., 9:15]), self._canonical_6d(primitive[..., 15:21])), -1)
        return {
            "primitive": primitive,
            "phase_logits": self.phase_head(final),
            "exit_logits": self.exit_head(final),
            "hidden": hidden_out,
        }

    @staticmethod
    def _canonical_6d(value: torch.Tensor) -> torch.Tensor:
        matrix = rotation_6d_to_matrix(value)
        return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), -1)

    def freeze_cnn(self) -> None:
        self.visual_encoder.requires_grad_(False)

    def unfreeze_last_block(self) -> None:
        self.visual_encoder.requires_grad_(False)
        convs = [m for m in self.visual_encoder.convs if isinstance(m, nn.Conv2d)]
        convs[-1].requires_grad_(True)
        self.visual_encoder.projection.requires_grad_(True)
