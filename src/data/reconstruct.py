from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
import minari
import mujoco
import numpy as np
from tqdm import tqdm

from src.envs.adroit import CAMERA_CONFIG, full_state, make_adroit_env
from src.utils import package_versions, save_json


def _metadata(dataset, episode_id: int) -> dict[str, Any]:
    return next(dataset.storage.get_episode_metadata([episode_id]))


def _save_contact_sheet(path: Path, images: list[np.ndarray]) -> None:
    rows = []
    for start in range(0, len(images), 5):
        row = np.concatenate(images[start : start + 5], axis=1)
        if row.shape[1] < images[0].shape[1] * 5:
            pad = np.zeros(
                (row.shape[0], images[0].shape[1] * 5 - row.shape[1], 3),
                dtype=np.uint8,
            )
            row = np.concatenate([row, pad], axis=1)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def build_visual_dataset(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    os.environ["MINARI_DATASETS_PATH"] = str(Path(config["dataset_path"]).resolve())
    output_path = Path(config["output_path"])
    if output_path.exists() and not force:
        validation_path = Path(config["validation_path"])
        if not validation_path.exists():
            raise FileExistsError(
                f"{output_path} exists without {validation_path}; pass --force to rebuild"
            )
        with h5py.File(output_path, "r") as existing:
            expected = {
                "dataset_id": config["dataset_id"],
                "environment_id": config["environment_id"],
                "horizon": int(config["horizon"]),
                "image_size": int(config["image_size"]),
                "frame_stack": int(config["frame_stack"]),
            }
            mismatched = {
                key: (existing.attrs.get(key), value)
                for key, value in expected.items()
                if existing.attrs.get(key) != value
            }
        if mismatched:
            raise RuntimeError(
                f"Existing processed dataset does not match config: {mismatched}; "
                "pass --force to rebuild"
            )
        with validation_path.open("r", encoding="utf-8") as handle:
            validation = json.load(handle)
        if not validation.get("passed", False):
            raise RuntimeError(f"Existing validation did not pass: {validation_path}")
        return validation
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = minari.load_dataset(config["dataset_id"], download=True)
    if dataset.spec.env_spec.id != config["environment_id"]:
        raise RuntimeError("Dataset environment does not match configured environment")
    if dataset.spec.env_spec.max_episode_steps != config["horizon"]:
        raise RuntimeError("Dataset horizon does not match configured horizon")

    env = make_adroit_env(
        render=True,
        image_size=int(config["image_size"]),
        horizon=int(config["horizon"]),
        reward_type=config["reward_type"],
    )
    rng = np.random.default_rng(int(config["split_seed"]))
    episode_ids = np.arange(dataset.total_episodes)
    rng.shuffle(episode_ids)
    train_ids = sorted(episode_ids[: int(config["train_episodes"])].tolist())
    validation_ids = sorted(episode_ids[int(config["train_episodes"]) :].tolist())

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    state_errors: list[float] = []
    replay_observation_errors: list[float] = []
    replay_reward_errors: list[float] = []
    sample_frames: list[np.ndarray] = []

    with h5py.File(temporary_path, "w") as out:
        out.attrs.update(
            {
                "dataset_id": config["dataset_id"],
                "environment_id": config["environment_id"],
                "reward_type": config["reward_type"],
                "horizon": int(config["horizon"]),
                "image_size": int(config["image_size"]),
                "frame_stack": int(config["frame_stack"]),
                "source_states": int(dataset.total_steps),
                "valid_transitions": int(dataset.total_steps - dataset.total_episodes),
                "action_dim": int(dataset.action_space.shape[0]),
                "camera_json": json.dumps(
                    {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in CAMERA_CONFIG.items()
                    }
                ),
                "renderer": config["renderer"],
                "train_episode_ids": np.asarray(train_ids, dtype=np.int32),
                "validation_episode_ids": np.asarray(validation_ids, dtype=np.int32),
            }
        )
        episodes_group = out.create_group("episodes")
        for ep in tqdm(dataset.iterate_episodes(), total=dataset.total_episodes, desc="replay"):
            metadata = _metadata(dataset, ep.id)
            obs, _ = env.reset(options=metadata["options"])
            qpos = np.empty((config["horizon"], 30), dtype=np.float64)
            qvel = np.empty((config["horizon"], 30), dtype=np.float64)
            desired = np.empty((config["horizon"], 4), dtype=np.float64)
            rgb = np.empty(
                (config["horizon"], config["image_size"], config["image_size"], 3),
                dtype=np.uint8,
            )
            replay_obs = np.empty((config["horizon"], 45), dtype=np.float64)
            replay_rewards = np.empty(config["horizon"], dtype=np.float64)

            for t, action in enumerate(ep.actions):
                state = full_state(env)
                qpos[t], qvel[t], desired[t] = (
                    state["qpos"],
                    state["qvel"],
                    state["desired_orien"],
                )
                # MuJoCo's Euler mj_step leaves some derived kinematics at the
                # final substep before qpos integration. The original Minari
                # observation therefore differs slightly from a state restored
                # with the documented forward pass. Explicitly forward here so
                # RGB and stored low-D observations correspond to qpos/qvel.
                mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
                replay_obs[t] = env.unwrapped._get_obs()
                rgb[t] = env.render()
                replay_observation_errors.append(
                    float(np.max(np.abs(replay_obs[t] - ep.observations[t])))
                )
                obs, reward, _, _, _ = env.step(action)
                replay_rewards[t] = reward
                replay_reward_errors.append(float(abs(reward - ep.rewards[t])))

            group = episodes_group.create_group(str(ep.id))
            compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
            group.create_dataset("rgb", data=rgb, chunks=(1, *rgb.shape[1:]), **compression)
            group.create_dataset("qpos", data=qpos, **compression)
            group.create_dataset("qvel", data=qvel, **compression)
            group.create_dataset("desired_orien", data=desired, **compression)
            group.create_dataset("observation", data=replay_obs, **compression)
            group.create_dataset("action", data=ep.actions, **compression)
            group.create_dataset("reward", data=replay_rewards, **compression)
            group.create_dataset("success", data=ep.infos["success"][: config["horizon"]])
            group.attrs["split"] = "train" if ep.id in train_ids else "validation"
            group.attrs["source_length"] = config["horizon"]
            group.attrs["transition_length"] = config["horizon"] - 1

            if len(sample_frames) < 10:
                for idx in np.linspace(0, config["horizon"] - 1, 10, dtype=int):
                    if len(sample_frames) >= 10:
                        break
                    sample_frames.append(rgb[idx])

        # Restore randomly selected stored states and compare to recorded low-D obs.
        choices = [
            (int(ep), int(t))
            for ep in range(dataset.total_episodes)
            for t in range(config["horizon"])
        ]
        for flat_idx in rng.choice(
            len(choices), size=int(config["validation_state_samples"]), replace=False
        ):
            ep_id, t = choices[int(flat_idx)]
            group = episodes_group[str(ep_id)]
            state = {
                "qpos": group["qpos"][t],
                "qvel": group["qvel"][t],
                "desired_orien": group["desired_orien"][t],
            }
            env.reset(options={"initial_state_dict": state})
            restored = env.unwrapped._get_obs()
            state_errors.append(
                float(np.max(np.abs(restored - group["observation"][t])))
            )

        # Independently step sampled legal intra-episode transitions.
        transition_choices = [
            (ep, t)
            for ep in range(dataset.total_episodes)
            for t in range(config["horizon"] - 1)
        ]
        step_qpos_errors: list[float] = []
        step_qvel_errors: list[float] = []
        step_reward_errors: list[float] = []
        for flat_idx in rng.choice(
            len(transition_choices),
            size=int(config["validation_step_samples"]),
            replace=False,
        ):
            ep_id, t = transition_choices[int(flat_idx)]
            group = episodes_group[str(ep_id)]
            env.reset(
                options={
                    "initial_state_dict": {
                        "qpos": group["qpos"][t],
                        "qvel": group["qvel"][t],
                        "desired_orien": group["desired_orien"][t],
                    }
                }
            )
            _, reward, _, _, _ = env.unwrapped.step(group["action"][t])
            next_state = full_state(env)
            step_qpos_errors.append(
                float(np.max(np.abs(next_state["qpos"] - group["qpos"][t + 1])))
            )
            step_qvel_errors.append(
                float(np.max(np.abs(next_state["qvel"] - group["qvel"][t + 1])))
            )
            step_reward_errors.append(float(abs(reward - group["reward"][t])))

    temporary_path.replace(output_path)
    env.close()
    _save_contact_sheet(Path(config["sample_dir"]) / "offline_states.png", sample_frames)

    validation = {
        "dataset_id": config["dataset_id"],
        "environment_id": config["environment_id"],
        "environment_xml": "gymnasium_robotics/envs/assets/adroit_hand/adroit_pen.xml",
        "versions": package_versions(),
        "episodes": int(dataset.total_episodes),
        "source_states": int(dataset.total_steps),
        "container_observations_including_replayed_final": int(
            sum(len(ep.observations) for ep in dataset.iterate_episodes())
        ),
        "valid_intra_episode_transitions": int(dataset.total_steps - dataset.total_episodes),
        "nominal_horizon": int(config["horizon"]),
        "action_dim": int(dataset.action_space.shape[0]),
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
        "camera": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in CAMERA_CONFIG.items()
        },
        "image_size": [int(config["image_size"]), int(config["image_size"])],
        "renderer": config["renderer"],
        "reward_type": config["reward_type"],
        "action_scaling": "Gymnasium normalized [-1,1] mapped once by AdroitHandPenEnv",
        "checks": {
            "restored_observation_samples": int(config["validation_state_samples"]),
            "restored_observation_max_abs_error": max(state_errors),
            "sequential_replay_observation_max_abs_error": max(replay_observation_errors),
            "sequential_replay_reward_max_abs_error": max(replay_reward_errors),
            "next_rgb_exact_by_episode_storage": True,
            "step_samples": int(config["validation_step_samples"]),
            "step_qpos_max_abs_error": max(step_qpos_errors),
            "step_qvel_max_abs_error": max(step_qvel_errors),
            "step_reward_max_abs_error": max(step_reward_errors),
            "cross_episode_transitions": 0,
        },
    }
    validation["passed"] = bool(
        validation["checks"]["restored_observation_max_abs_error"]
        <= float(config["restored_observation_atol"])
        and validation["checks"]["sequential_replay_observation_max_abs_error"]
        <= float(config["source_observation_atol"])
        and validation["checks"]["step_qpos_max_abs_error"]
        <= float(config["state_atol"])
        and validation["checks"]["step_qvel_max_abs_error"]
        <= float(config["state_atol"])
        and validation["checks"]["step_reward_max_abs_error"]
        <= float(config["reward_atol"])
    )
    save_json(config["validation_path"], validation)
    if not validation["passed"]:
        raise RuntimeError(f"Data validation failed; inspect {config['validation_path']}")
    return validation
