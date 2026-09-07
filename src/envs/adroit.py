from __future__ import annotations

import os
from collections import deque
from typing import Any, Literal

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import gymnasium_robotics
import numpy as np


ENV_ID = "AdroitHandPen-v1"
HORIZON = 200
ACTION_DIM = 24
ORACLE_DIM = 45
HAND_DIM = 24
CAMERA_CONFIG: dict[str, Any] = {
    "camera_id": -1,
    "distance": 1.0,
    "azimuth": -45.0,
    "elevation": -25.0,
    "lookat": np.asarray([0.0, -0.2, 0.22], dtype=np.float64),
}

gym.register_envs(gymnasium_robotics)


def make_adroit_env(
    *,
    render: bool = False,
    image_size: int = 84,
    horizon: int = HORIZON,
    reward_type: str = "dense",
) -> gym.Env:
    kwargs: dict[str, Any] = {
        "reward_type": reward_type,
        "max_episode_steps": horizon,
    }
    if render:
        kwargs.update(
            render_mode="rgb_array",
            width=image_size,
            height=image_size,
            camera_id=CAMERA_CONFIG["camera_id"],
        )
    env = gym.make(ENV_ID, **kwargs)
    if render:
        env.unwrapped.mujoco_renderer.default_cam_config = {
            "distance": CAMERA_CONFIG["distance"],
            "azimuth": CAMERA_CONFIG["azimuth"],
            "elevation": CAMERA_CONFIG["elevation"],
            "lookat": CAMERA_CONFIG["lookat"].copy(),
        }
    return env


def full_state(env: gym.Env) -> dict[str, np.ndarray]:
    state = env.unwrapped.get_env_state()
    return {key: np.asarray(value).copy() for key, value in state.items()}


def hand_proprio(env: gym.Env) -> np.ndarray:
    state = full_state(env)
    return np.concatenate([state["qpos"][:HAND_DIM], state["qvel"][:HAND_DIM]]).astype(
        np.float32
    )


def privileged_observation(env: gym.Env) -> np.ndarray:
    return np.asarray(env.unwrapped._get_obs(), dtype=np.float32).copy()


def current_metrics(env: gym.Env) -> dict[str, float | bool]:
    base = env.unwrapped
    object_pos = base.data.xpos[base.obj_body_id].ravel().copy()
    desired_pos = base.data.site_xpos[base.eps_ball_site_id].ravel().copy()
    object_orien = (
        base.data.site_xpos[base.obj_t_site_id]
        - base.data.site_xpos[base.obj_b_site_id]
    ) / base.pen_length
    desired_orien = (
        base.data.site_xpos[base.tar_t_site_id]
        - base.data.site_xpos[base.tar_b_site_id]
    ) / base.tar_length
    position_error = float(np.linalg.norm(object_pos - desired_pos))
    orientation_similarity = float(np.dot(object_orien, desired_orien))
    return {
        "position_error": position_error,
        "orientation_similarity": orientation_similarity,
        "orientation_error": float(1.0 - np.clip(orientation_similarity, -1.0, 1.0)),
        "official_goal": bool(position_error < 0.075 and orientation_similarity > 0.95),
        "dropped": bool(object_pos[2] < 0.075),
    }


Intervention = Literal[
    "normal",
    "black",
    "episode_shuffle",
    "temporal_shuffle",
    "target_occlusion",
]


class VisualAdroitEnv:
    """Four-frame visual actor input plus privileged critic state.

    Actor fields are exactly RGB history, qpos[:24], qvel[:24], and previous
    normalized action. The 45D simulator observation is returned separately for
    training-only critics and values.
    """

    def __init__(
        self,
        image_size: int = 84,
        frame_stack: int = 4,
        horizon: int = HORIZON,
        intervention: Intervention = "normal",
    ):
        self.env = make_adroit_env(
            render=True, image_size=image_size, horizon=horizon, reward_type="dense"
        )
        self.visual_source_env = (
            make_adroit_env(
                render=True, image_size=image_size, horizon=horizon, reward_type="dense"
            )
            if intervention == "episode_shuffle"
            else None
        )
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.intervention = intervention
        self.frames: deque[np.ndarray] = deque(maxlen=frame_stack)
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._rng = np.random.default_rng(0)

    def _transform_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.uint8).copy()
        if self.intervention == "black":
            frame.fill(0)
        elif self.intervention == "target_occlusion":
            # Mask green target pixels and a small local neighborhood without
            # disclosing any simulator target state to the actor.
            import cv2

            red, green, blue = frame[..., 0], frame[..., 1], frame[..., 2]
            mask = (
                (green > 60)
                & (green.astype(np.int16) > red.astype(np.int16) + 20)
                & (green.astype(np.int16) > blue.astype(np.int16) + 10)
            ).astype(np.uint8)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
            frame[mask] = 0
        return frame

    def _observation(self) -> dict[str, np.ndarray]:
        stacked = np.stack(tuple(self.frames), axis=0)
        if self.intervention == "temporal_shuffle":
            stacked = stacked[self._rng.permutation(self.frame_stack)]
        return {
            "rgb": stacked.transpose(0, 3, 1, 2).copy(),
            "proprio": hand_proprio(self.env),
            "previous_action": self.previous_action.copy(),
            "privileged": privileged_observation(self.env),
        }

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _, info = self.env.reset(seed=seed, options=options)
        self._rng = np.random.default_rng(seed)
        self.previous_action.fill(0)
        if self.visual_source_env is not None:
            source_seed = None if seed is None else int(seed) + 1_000_003
            self.visual_source_env.reset(seed=source_seed)
            frame = self._transform_frame(self.visual_source_env.render())
        else:
            frame = self._transform_frame(self.env.render())
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame.copy())
        return self._observation(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite (24,) normalized vector")
        action = np.clip(action, -1.0, 1.0)
        _, reward, terminated, truncated, info = self.env.step(action)
        if self.visual_source_env is not None:
            self.visual_source_env.step(action)
        self.previous_action = action.copy()
        visual_source = self.visual_source_env or self.env
        self.frames.append(self._transform_frame(visual_source.render()))
        info = dict(info)
        info.update(current_metrics(self.env))
        return self._observation(), float(reward), terminated, truncated, info

    def training_state(self) -> dict[str, Any]:
        """Capture the exact state needed to resume an online rollout."""
        state: dict[str, Any] = {
            "simulator": full_state(self.env),
            "elapsed_steps": int(getattr(self.env, "_elapsed_steps", 0)),
            "previous_action": self.previous_action.copy(),
            "frames": np.stack(tuple(self.frames), axis=0),
            "visual_rng": self._rng.bit_generator.state,
            "environment_rng": self.unwrapped.np_random.bit_generator.state,
        }
        return state

    def restore_training_state(self, state: dict[str, Any]) -> dict[str, np.ndarray]:
        """Restore a state captured by :meth:`training_state`."""
        simulator = {
            key: np.asarray(value).copy()
            for key, value in state["simulator"].items()
        }
        self.unwrapped.set_env_state(simulator)
        if hasattr(self.env, "_elapsed_steps"):
            self.env._elapsed_steps = int(state["elapsed_steps"])
        self.previous_action = np.asarray(
            state["previous_action"], dtype=np.float32
        ).copy()
        self.frames.clear()
        for frame in np.asarray(state["frames"], dtype=np.uint8):
            self.frames.append(frame.copy())
        self._rng.bit_generator.state = state["visual_rng"]
        self.unwrapped.np_random.bit_generator.state = state["environment_rng"]
        return self._observation()

    def close(self) -> None:
        self.env.close()
        if self.visual_source_env is not None:
            self.visual_source_env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped
