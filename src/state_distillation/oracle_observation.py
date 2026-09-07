from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ORACLE_DIM = 45
PRIMITIVE_DIM = 21
DESIRED_POSITION = (0.0, -0.2, 0.25)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Column-major Zhou 6D representation, explicit to avoid convention drift."""
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 6:
        raise ValueError("rotation 6D values must end in dimension 6")
    first = F.normalize(value[..., :3], dim=-1, eps=1e-6)
    second_raw = value[..., 3:]
    second = F.normalize(
        second_raw - (first * second_raw).sum(-1, keepdim=True) * first,
        dim=-1,
        eps=1e-6,
    )
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def geodesic_angle(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0 + 1e-6, 1.0 - 1e-6
    )
    return torch.acos(cosine)


@dataclass
class OraclePrimitiveState:
    object_position: torch.Tensor
    object_linear_velocity: torch.Tensor
    object_angular_velocity: torch.Tensor
    object_rotation_6d: torch.Tensor
    target_rotation_6d: torch.Tensor

    def vector(self) -> torch.Tensor:
        return torch.cat(
            (
                self.object_position,
                self.object_linear_velocity,
                self.object_angular_velocity,
                self.object_rotation_6d,
                self.target_rotation_6d,
            ),
            dim=-1,
        )

    @classmethod
    def from_vector(cls, value: torch.Tensor) -> "OraclePrimitiveState":
        if value.shape[-1] != PRIMITIVE_DIM:
            raise ValueError(f"primitive state must have {PRIMITIVE_DIM} values")
        return cls(value[..., 0:3], value[..., 3:6], value[..., 6:9], value[..., 9:15], value[..., 15:21])


class OracleObservationAdapter(nn.Module):
    """The sole deployable-known + predicted-primitive -> raw Oracle adapter.

    Adroit's 45D observation is world-frame hand qpos[0:24], object position,
    world-frame linear/angular velocity, the world direction of the pen and
    target local +Z axes, followed by two deterministic differences. The target
    position is a fixed task input, never inferred from simulator state here.
    """

    uses_privileged_state = False

    def __init__(self, desired_position: tuple[float, float, float] = DESIRED_POSITION):
        super().__init__()
        self.register_buffer("desired_position", torch.tensor(desired_position, dtype=torch.float32))

    def forward(self, proprio: torch.Tensor, primitive: torch.Tensor) -> torch.Tensor:
        if proprio.shape[-1] != 48:
            raise ValueError("deployable proprio must be 48D qpos/qvel")
        state = OraclePrimitiveState.from_vector(primitive)
        object_rotation = rotation_6d_to_matrix(state.object_rotation_6d)
        target_rotation = rotation_6d_to_matrix(state.target_rotation_6d)
        object_axis = object_rotation[..., :, 2]
        target_axis = target_rotation[..., :, 2]
        desired_position = self.desired_position.to(proprio).expand_as(state.object_position)
        return torch.cat(
            (
                proprio[..., :24],
                state.object_position,
                state.object_linear_velocity,
                state.object_angular_velocity,
                object_axis,
                target_axis,
                state.object_position - desired_position,
                object_axis - target_axis,
            ),
            dim=-1,
        )

    @staticmethod
    def primitive_from_runtime(env) -> np.ndarray:
        base = env.unwrapped
        observation = np.asarray(base._get_obs(), dtype=np.float32)
        object_rotation = np.asarray(base.data.xmat[base.obj_body_id], dtype=np.float32).reshape(3, 3)
        target_rotation = np.asarray(base.data.xmat[base.target_obj_body_id], dtype=np.float32).reshape(3, 3)
        value = np.concatenate(
            (
                observation[24:27],
                observation[27:30],
                observation[30:33],
                object_rotation[:, :2].T.reshape(-1),
                target_rotation[:, :2].T.reshape(-1),
            )
        ).astype(np.float32)
        # matrix_to_rotation_6d concatenates columns; transpose/reshape does that.
        return value

    @staticmethod
    def schema() -> list[dict[str, object]]:
        return [
            {"range": [0, 24], "name": "hand_qpos", "source": "deployable proprio", "frame": "joint coordinates"},
            {"range": [24, 27], "name": "object_position", "source": "predicted primitive", "frame": "world", "unit": "m"},
            {"range": [27, 30], "name": "object_linear_velocity", "source": "predicted primitive", "frame": "world", "unit": "m/s"},
            {"range": [30, 33], "name": "object_angular_velocity", "source": "predicted primitive", "frame": "world", "unit": "rad/s"},
            {"range": [33, 36], "name": "object_local_z_in_world", "source": "derived from predicted SO(3)"},
            {"range": [36, 39], "name": "target_local_z_in_world", "source": "derived from predicted SO(3)"},
            {"range": [39, 42], "name": "object_minus_fixed_target_position", "source": "deterministic derived"},
            {"range": [42, 45], "name": "object_axis_minus_target_axis", "source": "deterministic derived"},
        ]
