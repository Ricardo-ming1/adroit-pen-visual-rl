from __future__ import annotations

import copy
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.data.offline import NormalizationStats, PenOfflineData
from src.envs.adroit import HORIZON, VisualAdroitEnv, current_metrics
from src.evaluation import policy_inputs
from src.models.actor import SquashedGaussianActor
from src.models.safe_residual import headroom_action


ROOT_TYPES = (
    "far_pre_goal_failure",
    "near_goal",
    "goal_entry_pending_exit",
    "stable_hold",
)


def load_frozen_actor(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[SquashedGaussianActor, NormalizationStats, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    actor = SquashedGaussianActor(**checkpoint["actor_init"]).to(device)
    actor.load_state_dict(checkpoint["actor_state"])
    actor.eval().requires_grad_(False)
    return actor, NormalizationStats.from_dict(checkpoint["normalization"]), checkpoint


def clone_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in observation.items()}


@torch.inference_mode()
def frozen_visual_latent(
    actor: SquashedGaussianActor,
    observation: dict[str, np.ndarray],
    stats: NormalizationStats,
    device: torch.device,
) -> np.ndarray:
    inputs = policy_inputs(observation, "vision", stats, device)
    latent = actor.visual_latent(inputs["rgb"])
    return latent.squeeze(0).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def deterministic_action(
    actor: SquashedGaussianActor,
    line: str,
    observation: dict[str, np.ndarray],
    stats: NormalizationStats,
    device: torch.device,
) -> np.ndarray:
    action = actor.act(
        **policy_inputs(observation, line, stats, device), deterministic=True
    )
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def _near_goal(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(
        not metrics["official_goal"]
        and not metrics["dropped"]
        and float(metrics["position_error"])
        <= float(config["roots"]["near_position_error"])
        and float(metrics["orientation_similarity"])
        >= float(config["roots"]["near_orientation_similarity"])
    )


def _root_type_candidates(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, list[int]]:
    minimum_remaining = int(config["roots"]["minimum_remaining_steps"])
    exit_lookahead = int(config["roots"]["exit_lookahead"])
    candidates = {name: [] for name in ROOT_TYPES}
    state_goal = [bool(record["root_metrics"]["official_goal"]) for record in records]
    for index, record in enumerate(records):
        remaining = len(records) - index
        if remaining < minimum_remaining:
            continue
        metrics = record["root_metrics"]
        if metrics["dropped"]:
            continue
        future = state_goal[index + 1 : index + 1 + exit_lookahead]
        exits_soon = bool(metrics["official_goal"] and future and not all(future))
        if exits_soon:
            candidates["goal_entry_pending_exit"].append(index)
        if (
            metrics["official_goal"]
            and int(record["prefix"]["current_goal_streak"]) >= 20
            and all(state_goal[index : index + min(20, remaining)])
        ):
            candidates["stable_hold"].append(index)
        near = _near_goal(metrics, config)
        if near:
            candidates["near_goal"].append(index)
        if not metrics["official_goal"] and not near:
            future_goal = any(state_goal[index : index + min(10, remaining)])
            if not future_goal:
                candidates["far_pre_goal_failure"].append(index)
    return candidates


def _choose_root_index(
    root_type: str,
    indices: list[int],
    records: list[dict[str, Any]],
    rng: np.random.Generator,
) -> int:
    if root_type == "goal_entry_pending_exit":
        return min(indices)
    if root_type == "stable_hold":
        return max(
            indices,
            key=lambda index: int(records[index]["prefix"]["current_goal_streak"]),
        )
    if root_type == "near_goal":
        ordered = sorted(
            indices,
            key=lambda index: (
                -float(records[index]["root_metrics"]["orientation_similarity"]),
                float(records[index]["root_metrics"]["position_error"]),
            ),
        )
        return ordered[min(len(ordered) - 1, int(rng.integers(0, min(5, len(ordered)))))]
    middle = [index for index in indices if index >= 20]
    pool = middle or indices
    return int(pool[int(rng.integers(0, len(pool)))])


def _event_metrics(
    root: dict[str, Any],
    official: list[bool],
    dropped: list[bool],
    rewards: list[float],
    gamma: float,
    complete: bool,
) -> dict[str, Any]:
    prefix = root["prefix"]
    ever_goal = bool(prefix["ever_goal"])
    current_streak = int(prefix["current_goal_streak"])
    maximum = current_streak
    entered_after_root = False
    local_exit = False
    active_entry = bool(root["root_metrics"]["official_goal"])
    goal_steps = int(active_entry)
    for reached in official:
        reached = bool(reached)
        if reached and not ever_goal:
            entered_after_root = True
        ever_goal = ever_goal or reached
        if active_entry and not reached:
            local_exit = True
        active_entry = active_entry or reached
        current_streak = current_streak + 1 if reached else 0
        maximum = max(maximum, current_streak)
        goal_steps += int(reached)
    discounted_return = 0.0
    discount = 1.0
    for reward in rewards:
        discounted_return += discount * float(reward)
        discount *= float(gamma)
    strict = bool(
        complete
        and len(official) >= 20
        and all(official[-20:])
        and not bool(prefix["dropped"])
        and not any(dropped)
    )
    return {
        "drop": bool(prefix["dropped"] or any(dropped)),
        "benchmark": ever_goal,
        "entry_after_root": entered_after_root,
        "goal_steps": int(goal_steps),
        "max_goal_streak": int(maximum),
        "goal_exit_after_root": local_exit,
        "strict": strict,
        "discounted_return": float(discounted_return),
        "steps": len(rewards),
    }


def _source_outcome(
    root_record_index: int,
    records: list[dict[str, Any]],
    root: dict[str, Any],
    gamma: float,
) -> dict[str, Any]:
    continuation = records[root_record_index:]
    return _event_metrics(
        root,
        [bool(record["post_info"]["official_goal"]) for record in continuation],
        [bool(record["post_info"]["dropped"]) for record in continuation],
        [float(record["reward"]) for record in continuation],
        gamma,
        complete=True,
    )


def _make_root(
    *,
    root_id: str,
    root_type: str,
    record_index: int,
    records: list[dict[str, Any]],
    episode_seed: int,
    episode_index: int,
    gamma: float,
    verify_steps: int,
) -> dict[str, Any]:
    record = records[record_index]
    root = {
        "root_id": root_id,
        "root_type": root_type,
        "source_episode_seed": int(episode_seed),
        "source_episode_index": int(episode_index),
        "elapsed_step": int(record["elapsed_step"]),
        "remaining_steps": int(len(records) - record_index),
        "state": copy.deepcopy(record["state"]),
        "source_observation": clone_observation(record["observation"]),
        "root_metrics": copy.deepcopy(record["root_metrics"]),
        "prefix": copy.deepcopy(record["prefix"]),
        "proprio_history": np.asarray(record["proprio_history"], dtype=np.float32),
        "frozen_visual_latent_history": np.asarray(
            record["frozen_visual_latent_history"], dtype=np.float32
        ),
        "previous_base_actions": np.asarray(
            record["previous_base_actions"], dtype=np.float32
        ),
        "previous_executed_actions": np.asarray(
            record["previous_executed_actions"], dtype=np.float32
        ),
    }
    root["source_outcome"] = _source_outcome(record_index, records, root, gamma)
    root["source_verification"] = [
        {
            "action": entry["action"].copy(),
            "reward": float(entry["reward"]),
            "terminated": bool(entry["terminated"]),
            "truncated": bool(entry["truncated"]),
            "next_observation": clone_observation(entry["next_observation"]),
            "official_goal": bool(entry["post_info"]["official_goal"]),
            "dropped": bool(entry["post_info"]["dropped"]),
        }
        for entry in records[
            record_index : min(len(records), record_index + verify_steps)
        ]
    ]
    return root


@torch.inference_mode()
def collect_training_roots(
    *,
    vision_actor: SquashedGaussianActor,
    stats: NormalizationStats,
    device: torch.device,
    config: dict[str, Any],
    existing_roots: list[dict[str, Any]] | None = None,
    start_episode_index: int = 0,
    checkpoint_callback: Callable[[list[dict[str, Any]], int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = {key: int(config["roots"]["quotas"][key]) for key in ROOT_TYPES}
    roots = list(existing_roots or [])
    counts = Counter(root["root_type"] for root in roots)
    episode_seed_start = int(config["roots"]["training_seed_start"])
    maximum_episodes = int(config["roots"]["maximum_source_episodes"])
    gamma = float(config["branches"]["gamma"])
    verify_steps = int(config["roots"]["zero_verify_steps"])
    env = VisualAdroitEnv()
    scanned = start_episode_index

    for episode_index in range(start_episode_index, maximum_episodes):
        if all(counts[key] >= quotas[key] for key in ROOT_TYPES):
            break
        episode_seed = episode_seed_start + episode_index
        episode_rng = np.random.default_rng(int(config["seed"]) * 1_000_003 + episode_seed)
        observation, _ = env.reset(seed=episode_seed)
        records: list[dict[str, Any]] = []
        previous_base = deque(
            [np.zeros(24, dtype=np.float32) for _ in range(4)], maxlen=4
        )
        previous_executed = deque(
            [np.zeros(24, dtype=np.float32) for _ in range(4)], maxlen=4
        )
        proprio_history = deque(
            [observation["proprio"].copy() for _ in range(4)], maxlen=4
        )
        initial_latent = frozen_visual_latent(
            vision_actor, observation, stats, device
        )
        latent_history = deque(
            [initial_latent.copy() for _ in range(4)], maxlen=4
        )
        ever_goal = False
        dropped_prefix = False
        streak = 0

        for elapsed_step in range(HORIZON):
            root_metrics = current_metrics(env.env)
            ever_goal = ever_goal or bool(root_metrics["official_goal"])
            dropped_prefix = dropped_prefix or bool(root_metrics["dropped"])
            streak = streak + 1 if root_metrics["official_goal"] else 0
            proprio_history.append(observation["proprio"].copy())
            latent_history.append(
                frozen_visual_latent(vision_actor, observation, stats, device)
            )
            action = deterministic_action(
                vision_actor, "vision", observation, stats, device
            )
            state = env.training_state()
            next_observation, reward, terminated, truncated, info = env.step(action)
            records.append(
                {
                    "elapsed_step": elapsed_step,
                    "state": state,
                    "observation": clone_observation(observation),
                    "root_metrics": root_metrics,
                    "prefix": {
                        "ever_goal": ever_goal,
                        "dropped": dropped_prefix,
                        "current_goal_streak": streak,
                    },
                    "proprio_history": np.stack(tuple(proprio_history)),
                    "frozen_visual_latent_history": np.stack(tuple(latent_history)),
                    "previous_base_actions": np.stack(tuple(previous_base)),
                    "previous_executed_actions": np.stack(tuple(previous_executed)),
                    "action": action.copy(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "post_info": {
                        "official_goal": bool(info["official_goal"]),
                        "dropped": bool(info["dropped"]),
                    },
                    "next_observation": clone_observation(next_observation),
                }
            )
            previous_base.append(action.copy())
            previous_executed.append(action.copy())
            observation = next_observation
            if terminated or truncated:
                break

        available = _root_type_candidates(records, config)
        rare_priority = {
            "stable_hold": 0,
            "goal_entry_pending_exit": 1,
            "near_goal": 2,
            "far_pre_goal_failure": 3,
        }
        priority = sorted(
            ROOT_TYPES,
            key=lambda key: (
                counts[key] / max(quotas[key], 1),
                rare_priority[key],
            ),
        )
        selected = 0
        for root_type in priority:
            if selected >= int(config["roots"]["max_roots_per_trajectory"]):
                break
            if counts[root_type] >= quotas[root_type] or not available[root_type]:
                continue
            index = _choose_root_index(root_type, available[root_type], records, episode_rng)
            root_id = f"root_{len(roots):04d}_{root_type}"
            roots.append(
                _make_root(
                    root_id=root_id,
                    root_type=root_type,
                    record_index=index,
                    records=records,
                    episode_seed=episode_seed,
                    episode_index=episode_index,
                    gamma=gamma,
                    verify_steps=verify_steps,
                )
            )
            counts[root_type] += 1
            selected += 1
        scanned = episode_index + 1
        if checkpoint_callback is not None and scanned % int(
            config["roots"]["checkpoint_episode_interval"]
        ) == 0:
            checkpoint_callback(roots, scanned)

    env.close()
    complete = all(counts[key] >= quotas[key] for key in ROOT_TYPES)
    metadata = {
        "source_episodes_scanned": scanned,
        "source_rollout_transitions": scanned * HORIZON,
        "counts": {key: int(counts[key]) for key in ROOT_TYPES},
        "quotas": quotas,
        "complete": complete,
        "maximum_roots_per_source_trajectory": int(
            config["roots"]["max_roots_per_trajectory"]
        ),
        "training_seed_start": episode_seed_start,
    }
    if not complete:
        raise RuntimeError(f"Could not fill root quotas: {metadata}")
    return roots, metadata


@torch.inference_mode()
def verify_zero_branches(
    roots: list[dict[str, Any]],
    vision_actor: SquashedGaussianActor,
    stats: NormalizationStats,
    device: torch.device,
) -> dict[str, Any]:
    env = VisualAdroitEnv()
    env.reset(seed=0)
    maxima = Counter()
    mismatches = 0
    checked_steps = 0
    for root in roots:
        observation = env.restore_training_state(copy.deepcopy(root["state"]))
        for key, expected in root["source_observation"].items():
            maxima[f"initial_{key}_max_abs"] = max(
                maxima[f"initial_{key}_max_abs"],
                float(np.max(np.abs(observation[key].astype(np.float64) - expected))),
            )
        for expected in root["source_verification"]:
            action = deterministic_action(
                vision_actor, "vision", observation, stats, device
            )
            action_error = float(np.max(np.abs(action - expected["action"])))
            observation, reward, terminated, truncated, info = env.step(action)
            reward_error = abs(float(reward) - float(expected["reward"]))
            maxima["action_max_abs"] = max(maxima["action_max_abs"], action_error)
            maxima["reward_max_abs"] = max(maxima["reward_max_abs"], reward_error)
            for key, expected_value in expected["next_observation"].items():
                error = float(
                    np.max(
                        np.abs(
                            observation[key].astype(np.float64)
                            - np.asarray(expected_value, dtype=np.float64)
                        )
                    )
                )
                maxima[f"next_{key}_max_abs"] = max(
                    maxima[f"next_{key}_max_abs"], error
                )
            exact_events = bool(
                bool(terminated) == expected["terminated"]
                and bool(truncated) == expected["truncated"]
                and bool(info["official_goal"]) == expected["official_goal"]
                and bool(info["dropped"]) == expected["dropped"]
            )
            if action_error > 1e-6 or reward_error > 1e-9 or not exact_events:
                mismatches += 1
            checked_steps += 1
    env.close()
    return {
        "roots_checked": len(roots),
        "continuation_steps_checked": checked_steps,
        "mismatch_count": mismatches,
        "pass": mismatches == 0
        and maxima["action_max_abs"] < 1e-6
        and maxima["reward_max_abs"] < 1e-9,
        **{key: float(value) for key, value in maxima.items()},
    }


def offline_action_pca(data: PenOfflineData, components: int = 4) -> np.ndarray:
    axes = [index for index, split in enumerate(data.splits) if split == "train"]
    action = data.action[axes, :-1].reshape(-1, data.action_dim).astype(np.float64)
    action -= action.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(action, full_matrices=False)
    directions = right[:components]
    rms = np.sqrt(np.mean(np.square(directions), axis=1, keepdims=True))
    return (directions / np.maximum(rms, 1e-8)).astype(np.float32)


def candidate_specs(pca_directions: np.ndarray) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {"name": "zero", "kind": "zero", "intervention_steps": 0, "local": True},
        {
            "name": "safe_residual_mean",
            "kind": "safe_residual",
            "intervention_steps": 3,
            "local": True,
        },
    ]
    for gain in (0.10, 0.25, 0.50, 1.00):
        candidates.append(
            {
                "name": f"oracle_direction_gain_{gain:.2f}",
                "kind": "oracle_direction",
                "gain": gain,
                "intervention_steps": 3,
                "local": True,
            }
        )
    amplitudes = (0.0025, 0.0050, 0.0100, 0.0050)
    for index, (direction, amplitude) in enumerate(zip(pca_directions, amplitudes)):
        for sign in (-1.0, 1.0):
            candidates.append(
                {
                    "name": f"pca_{index}_{'pos' if sign > 0 else 'neg'}",
                    "kind": "pca",
                    "direction": (sign * direction).astype(np.float32),
                    "correction_rms_target": amplitude,
                    "intervention_steps": 3,
                    "local": True,
                }
            )
    candidates.extend(
        [
            {
                "name": "oracle_takeover_5",
                "kind": "oracle_takeover",
                "intervention_steps": 5,
                "local": False,
            },
            {
                "name": "oracle_takeover_20",
                "kind": "oracle_takeover",
                "intervention_steps": 20,
                "local": False,
            },
        ]
    )
    return candidates


def _candidate_json(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in spec.items()
    }


@torch.inference_mode()
def _candidate_action(
    *,
    spec: dict[str, Any],
    branch_step: int,
    observation: dict[str, np.ndarray],
    vision_actor: SquashedGaussianActor,
    vision_stats: NormalizationStats,
    oracle_actor: SquashedGaussianActor,
    oracle_stats: NormalizationStats,
    safe_policy: Any,
    device: torch.device,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = deterministic_action(
        vision_actor, "vision", observation, vision_stats, device
    )
    if branch_step >= int(spec["intervention_steps"]) or spec["kind"] == "zero":
        return base, np.zeros_like(base), base
    if spec["kind"] == "oracle_takeover":
        oracle = deterministic_action(
            oracle_actor, "oracle", observation, oracle_stats, device
        )
        return oracle, np.zeros_like(base), base
    if spec["kind"] == "safe_residual":
        inputs = policy_inputs(observation, "vision", vision_stats, device)
        action_tensor, info = safe_policy.action_and_info(
            deterministic=True, rho_effective=rho, **inputs
        )
        return (
            action_tensor.squeeze(0).cpu().numpy().astype(np.float32),
            info["residual_latent"].squeeze(0).cpu().numpy().astype(np.float32),
            base,
        )
    if spec["kind"] == "oracle_direction":
        oracle = deterministic_action(
            oracle_actor, "oracle", observation, oracle_stats, device
        )
        direction = oracle - base
        scale = max(float(np.max(np.abs(direction))), 1e-8)
        z = np.clip(float(spec["gain"]) * direction / scale, -1.0, 1.0)
    elif spec["kind"] == "pca":
        desired = float(spec["correction_rms_target"]) * np.asarray(
            spec["direction"], dtype=np.float32
        )
        room = np.where(desired >= 0.0, 1.0 - base, 1.0 + base)
        z = np.clip(desired / np.maximum(float(rho) * room, 1e-8), -1.0, 1.0)
    else:
        raise ValueError(f"Unknown candidate kind: {spec['kind']}")
    action, _ = headroom_action(
        torch.as_tensor(base, device=device).unsqueeze(0),
        torch.as_tensor(z, device=device).unsqueeze(0),
        rho,
    )
    return action.squeeze(0).cpu().numpy().astype(np.float32), z.astype(np.float32), base


def _rollout_segment(
    *,
    env: VisualAdroitEnv,
    root: dict[str, Any],
    spec: dict[str, Any],
    start_step: int,
    stop_step: int,
    observation: dict[str, np.ndarray],
    official: list[bool],
    dropped: list[bool],
    rewards: list[float],
    trace: list[dict[str, Any]],
    vision_actor: SquashedGaussianActor,
    vision_stats: NormalizationStats,
    oracle_actor: SquashedGaussianActor,
    oracle_stats: NormalizationStats,
    safe_policy: Any,
    device: torch.device,
    rho: float,
) -> tuple[dict[str, np.ndarray], bool]:
    finished = False
    for branch_step in range(start_step, stop_step):
        action, z, base = _candidate_action(
            spec=spec,
            branch_step=branch_step,
            observation=observation,
            vision_actor=vision_actor,
            vision_stats=vision_stats,
            oracle_actor=oracle_actor,
            oracle_stats=oracle_stats,
            safe_policy=safe_policy,
            device=device,
            rho=rho,
        )
        if branch_step < int(spec["intervention_steps"]):
            trace.append(
                {
                    "z": z.tolist(),
                    "base_action": base.tolist(),
                    "executed_action": action.tolist(),
                    "correction_rms": float(np.sqrt(np.mean(np.square(action - base)))),
                }
            )
        observation, reward, terminated, truncated, info = env.step(action)
        official.append(bool(info["official_goal"]))
        dropped.append(bool(info["dropped"]))
        rewards.append(float(reward))
        if terminated or truncated:
            finished = True
            break
    return observation, finished


def _screen_rank(metrics: dict[str, Any]) -> tuple:
    return (
        -int(metrics["drop"]),
        int(metrics["benchmark"]),
        int(metrics["max_goal_streak"]),
        -int(metrics["goal_exit_after_root"]),
        int(metrics["goal_steps"]),
        float(metrics["discounted_return"]),
    )


def _safe_relative(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    root_type: str,
) -> bool:
    if candidate["drop"] and not baseline["drop"]:
        return False
    if baseline["benchmark"] and not candidate["benchmark"]:
        return False
    if candidate["strict"] < baseline["strict"]:
        return False
    if candidate["max_goal_streak"] < baseline["max_goal_streak"]:
        return False
    if candidate["goal_steps"] < baseline["goal_steps"]:
        return False
    if root_type in ("goal_entry_pending_exit", "stable_hold") and (
        candidate["goal_exit_after_root"] > baseline["goal_exit_after_root"]
    ):
        return False
    return True


def _task_event_improved(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> bool:
    return bool(
        candidate["strict"] > baseline["strict"]
        or candidate["benchmark"] > baseline["benchmark"]
        or candidate["max_goal_streak"] > baseline["max_goal_streak"]
        or candidate["goal_exit_after_root"] < baseline["goal_exit_after_root"]
        or candidate["goal_steps"] > baseline["goal_steps"]
    )


def _full_rank(result: dict[str, Any]) -> tuple:
    metrics = result["full"]
    correction = result["correction_rms"]
    return (
        int(metrics["strict"]),
        int(metrics["benchmark"]),
        int(metrics["max_goal_streak"]),
        -int(metrics["goal_exit_after_root"]),
        int(metrics["goal_steps"]),
        float(metrics["discounted_return"]),
        -float(correction),
        -int(result["candidate"]["intervention_steps"]),
        int(result["candidate"]["name"] == "zero"),
    )


@torch.inference_mode()
def run_branch_pilot(
    *,
    roots: list[dict[str, Any]],
    vision_actor: SquashedGaussianActor,
    vision_stats: NormalizationStats,
    oracle_actor: SquashedGaussianActor,
    oracle_stats: NormalizationStats,
    safe_policy: Any,
    pca_directions: np.ndarray,
    device: torch.device,
    config: dict[str, Any],
    progress_callback: Callable[[list[dict[str, Any]], int], None] | None = None,
    existing_results: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = candidate_specs(pca_directions)
    screen_steps = int(config["branches"]["screen_steps"])
    gamma = float(config["branches"]["gamma"])
    rho = float(config["branches"]["rho"])
    top_k = int(config["branches"]["completion_top_k"])
    results = list(existing_results or [])
    completed_ids = {result["root_id"] for result in results}
    transition_count = sum(int(result["branch_transitions"]) for result in results)
    env = VisualAdroitEnv()
    env.reset(seed=0)

    for root_index, root in enumerate(roots):
        if root["root_id"] in completed_ids:
            continue
        screen_records: dict[str, dict[str, Any]] = {}
        for spec in specs:
            observation = env.restore_training_state(copy.deepcopy(root["state"]))
            official: list[bool] = []
            dropped: list[bool] = []
            rewards: list[float] = []
            trace: list[dict[str, Any]] = []
            observation, finished = _rollout_segment(
                env=env,
                root=root,
                spec=spec,
                start_step=0,
                stop_step=min(screen_steps, int(root["remaining_steps"])),
                observation=observation,
                official=official,
                dropped=dropped,
                rewards=rewards,
                trace=trace,
                vision_actor=vision_actor,
                vision_stats=vision_stats,
                oracle_actor=oracle_actor,
                oracle_stats=oracle_stats,
                safe_policy=safe_policy,
                device=device,
                rho=rho,
            )
            transition_count += len(rewards)
            screen_records[spec["name"]] = {
                "candidate": _candidate_json(spec),
                "screen": _event_metrics(
                    root, official, dropped, rewards, gamma, complete=finished
                ),
                "trace": trace,
                "state": copy.deepcopy(env.training_state()),
                "observation": clone_observation(observation),
                "official": official,
                "dropped": dropped,
                "rewards": rewards,
                "finished": finished,
            }

        zero_screen = screen_records["zero"]["screen"]
        local_nonzero = [
            record
            for name, record in screen_records.items()
            if record["candidate"]["local"] and name != "zero"
        ]
        eligible_screen = [
            record
            for record in local_nonzero
            if _safe_relative(record["screen"], zero_screen, root["root_type"])
        ]
        ranked = sorted(eligible_screen or local_nonzero, key=lambda value: _screen_rank(value["screen"]), reverse=True)
        completion_names = ["zero"] + [
            record["candidate"]["name"] for record in ranked[:top_k]
        ]
        completed: dict[str, dict[str, Any]] = {}
        for name in completion_names:
            record = screen_records[name]
            if not record["finished"]:
                env.restore_training_state(copy.deepcopy(record["state"]))
                before = len(record["rewards"])
                _, finished = _rollout_segment(
                    env=env,
                    root=root,
                    spec=record["candidate"],
                    start_step=before,
                    stop_step=int(root["remaining_steps"]),
                    observation=clone_observation(record["observation"]),
                    official=record["official"],
                    dropped=record["dropped"],
                    rewards=record["rewards"],
                    trace=record["trace"],
                    vision_actor=vision_actor,
                    vision_stats=vision_stats,
                    oracle_actor=oracle_actor,
                    oracle_stats=oracle_stats,
                    safe_policy=safe_policy,
                    device=device,
                    rho=rho,
                )
                transition_count += len(record["rewards"]) - before
                record["finished"] = finished
            corrections = [entry["correction_rms"] for entry in record["trace"]]
            completed[name] = {
                "candidate": record["candidate"],
                "screen": record["screen"],
                "full": _event_metrics(
                    root,
                    record["official"],
                    record["dropped"],
                    record["rewards"],
                    gamma,
                    complete=True,
                ),
                "trace": record["trace"],
                "correction_rms": float(np.mean(corrections) if corrections else 0.0),
            }

        zero = completed["zero"]
        local_completed = [
            value
            for name, value in completed.items()
            if name != "zero"
            and value["candidate"]["local"]
            and _safe_relative(value["full"], zero["full"], root["root_type"])
        ]
        winner = max([zero] + local_completed, key=_full_rank)
        positive = bool(
            winner["candidate"]["name"] != "zero"
            and _task_event_improved(winner["full"], zero["full"])
        )
        if not positive:
            winner = zero

        takeover_diagnostics = {}
        for name in ("oracle_takeover_5", "oracle_takeover_20"):
            record = screen_records[name]
            takeover_diagnostics[name] = {
                "screen": record["screen"],
                "screen_event_improved": bool(
                    _safe_relative(record["screen"], zero_screen, root["root_type"])
                    and _task_event_improved(record["screen"], zero_screen)
                ),
            }
        branch_transitions = sum(
            int(record["screen"]["steps"]) for record in screen_records.values()
        ) + sum(
            int(completed[name]["full"]["steps"] - completed[name]["screen"]["steps"])
            for name in completed
        )
        result = {
            "root_id": root["root_id"],
            "root_type": root["root_type"],
            "source_episode_seed": root["source_episode_seed"],
            "split": root.get("split"),
            "branch_transitions": branch_transitions,
            "zero": zero,
            "completed_candidates": completed,
            "takeover_diagnostics": takeover_diagnostics,
            "gate_target": int(positive),
            "residual_target": (
                winner["trace"][0]["z"] if positive and winner["trace"] else [0.0] * 24
            ),
            "winner": winner["candidate"]["name"],
            "winner_source": winner["candidate"]["kind"],
            "winner_correction_rms": winner["correction_rms"],
        }
        results.append(result)
        if progress_callback is not None and (root_index + 1) % int(
            config["branches"]["checkpoint_root_interval"]
        ) == 0:
            progress_callback(results, transition_count)

    env.close()
    counts = Counter(result["root_type"] for result in results)
    positives = Counter(
        result["root_type"] for result in results if result["gate_target"]
    )
    by_type = {
        root_type: {
            "roots": int(counts[root_type]),
            "positive_roots": int(positives[root_type]),
            "positive_root_rate": float(
                positives[root_type] / counts[root_type] if counts[root_type] else 0.0
            ),
        }
        for root_type in ROOT_TYPES
    }
    overall_rate = float(sum(positives.values()) / len(results))
    entry_exit_count = counts["goal_entry_pending_exit"]
    entry_exit_rate = float(
        positives["goal_entry_pending_exit"] / entry_exit_count
        if entry_exit_count
        else 0.0
    )
    winner_sources = Counter(
        result["winner_source"] for result in results if result["gate_target"]
    )
    takeover = {
        name: float(
            np.mean(
                [
                    result["takeover_diagnostics"][name]["screen_event_improved"]
                    for result in results
                ]
            )
        )
        for name in ("oracle_takeover_5", "oracle_takeover_20")
    }
    decision = (
        "proceed_student"
        if overall_rate >= 0.15 and entry_exit_rate >= 0.10
        else "extend_120_roots"
        if overall_rate >= 0.10
        else "stop_local_residual"
    )
    summary = {
        "roots": len(results),
        "local_candidates_including_zero_per_root": 14,
        "route_only_takeover_candidates_per_root": 2,
        "branch_transitions": int(sum(result["branch_transitions"] for result in results)),
        "overall_positive_roots": int(sum(positives.values())),
        "overall_positive_root_rate": overall_rate,
        "goal_entry_exit_positive_root_rate": entry_exit_rate,
        "by_root_type": by_type,
        "positive_winner_sources": dict(winner_sources),
        "takeover_screen_event_improvement_rate": takeover,
        "decision": decision,
    }
    return results, summary
