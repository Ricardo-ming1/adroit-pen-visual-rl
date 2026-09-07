from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Callable

import numpy as np
import torch

from src.counterfactual.branch_rollout import (
    _candidate_action,
    _candidate_json,
    _event_metrics,
    _rollout_segment,
)
from src.data.offline import NormalizationStats
from src.envs.adroit import VisualAdroitEnv
from src.models.actor import SquashedGaussianActor


ROOT_PHASES = (
    "far_pre_goal_failure",
    "near_goal",
    "goal_entry_pending_exit",
    "stable_hold",
)


def _kmeans(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Small deterministic k-means used only on branch-train correction chunks."""
    if points.ndim != 2 or len(points) < k:
        raise ValueError(f"Need at least {k} train chunks, got {points.shape}")
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=k, replace=False)].copy()
    for _ in range(100):
        distance = np.square(points[:, None] - centers[None]).sum(axis=-1)
        assignment = distance.argmin(axis=1)
        updated = np.stack(
            [points[assignment == i].mean(axis=0) if np.any(assignment == i)
             else points[int(rng.integers(len(points)))] for i in range(k)]
        )
        if np.max(np.abs(updated - centers)) < 1e-7:
            centers = updated
            break
        centers = updated
    return centers.astype(np.float32)


def build_train_codebook(
    v4_results: list[dict[str, Any]], size: int, seed: int
) -> np.ndarray:
    """Cluster only fixed PCA candidate chunks from train source trajectories.

    Candidate outcomes and Oracle-derived candidates are deliberately ignored, so the
    codebook cannot leak dev labels or privileged Oracle directions.
    """
    chunks: list[np.ndarray] = []
    for root in v4_results:
        if root.get("split") != "train":
            continue
        for record in root["completed_candidates"].values():
            if record["candidate"]["kind"] != "pca" or len(record["trace"]) != 3:
                continue
            chunks.append(np.asarray([x["z"] for x in record["trace"]], np.float32))
    centers = _kmeans(np.stack(chunks).reshape(len(chunks), -1), size, seed)
    return np.clip(centers.reshape(size, 3, 24), -1.0, 1.0)


def deployable_specs(pca: np.ndarray, codebook: np.ndarray) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"name": "zero", "kind": "zero", "intervention_steps": 0,
         "local": True, "deployable": True}
    ]
    amplitudes = (0.0025, 0.0050, 0.0100, 0.0050)
    for index, (direction, amplitude) in enumerate(zip(pca, amplitudes)):
        for sign in (-1.0, 1.0):
            specs.append({
                "name": f"pca_{index}_{'pos' if sign > 0 else 'neg'}",
                "kind": "pca",
                "direction": (sign * direction).astype(np.float32),
                "correction_rms_target": amplitude,
                "intervention_steps": 3,
                "local": True,
                "deployable": True,
            })
    for index, chunk in enumerate(codebook):
        specs.append({
            "name": f"codebook_{index}", "kind": "fixed_chunk",
            "z_chunk": chunk.astype(np.float32), "intervention_steps": 3,
            "local": True, "deployable": True,
        })
    return specs


def candidate_descriptor(spec: dict[str, Any]) -> np.ndarray:
    if spec["kind"] == "zero":
        return np.zeros((3, 24), np.float32)
    if spec["kind"] == "fixed_chunk":
        return np.asarray(spec["z_chunk"], np.float32)
    if spec["kind"] == "pca":
        direction = np.asarray(spec["direction"], np.float32)
        direction /= max(float(np.sqrt(np.mean(np.square(direction)))), 1e-8)
        # Descriptor is a fixed signed direction/amplitude. Executed correction still
        # uses the exact state-dependent headroom implementation below.
        z = np.clip(float(spec["correction_rms_target"]) * direction / 0.05, -1, 1)
        return np.repeat(z[None], 3, axis=0).astype(np.float32)
    raise ValueError(spec["kind"])


def _patched_spec_action(original):
    def action(**kwargs):
        spec = kwargs["spec"]
        if spec["kind"] != "fixed_chunk":
            return original(**kwargs)
        branch_step = int(kwargs["branch_step"])
        zero = dict(spec); zero["kind"] = "zero"; zero["intervention_steps"] = 0
        base, _, _ = original(**{**kwargs, "spec": zero})
        if branch_step >= int(spec["intervention_steps"]):
            return base, np.zeros_like(base), base
        from src.models.safe_residual import headroom_action
        z = np.asarray(spec["z_chunk"][branch_step], np.float32)
        action_tensor, _ = headroom_action(
            torch.as_tensor(base, device=kwargs["device"]).unsqueeze(0),
            torch.as_tensor(z, device=kwargs["device"]).unsqueeze(0),
            float(kwargs["rho"]),
        )
        return action_tensor.squeeze(0).cpu().numpy().astype(np.float32), z, base
    return action


_candidate_with_chunk = _patched_spec_action(_candidate_action)


@torch.inference_mode()
def run_common_horizon_branches(
    *, roots: list[dict[str, Any]], specs: list[dict[str, Any]],
    vision_actor: SquashedGaussianActor, vision_stats: NormalizationStats,
    oracle_actor: SquashedGaussianActor, oracle_stats: NormalizationStats,
    device: torch.device, rho: float, gamma: float,
    existing: list[dict[str, Any]] | None = None,
    callback: Callable[[list[dict[str, Any]], int], None] | None = None,
    checkpoint_interval: int = 4,
) -> tuple[list[dict[str, Any]], int]:
    # _rollout_segment calls its module-global action function. Patch it narrowly for
    # the duration of this synchronous data build, then restore it.
    import src.counterfactual.branch_rollout as rollout_module
    previous = rollout_module._candidate_action
    rollout_module._candidate_action = _candidate_with_chunk
    records = list(existing or [])
    done = {x["root_id"] for x in records}
    transitions = sum(int(x["branch_transitions"]) for x in records)
    env = VisualAdroitEnv(); env.reset(seed=0)
    try:
        for root_index, root in enumerate(roots):
            if root["root_id"] in done:
                continue
            candidates: list[dict[str, Any]] = []
            for candidate_index, spec in enumerate(specs):
                observation = env.restore_training_state(copy.deepcopy(root["state"]))
                official: list[bool] = []; dropped: list[bool] = []
                rewards: list[float] = []; trace: list[dict[str, Any]] = []
                _, finished = _rollout_segment(
                    env=env, root=root, spec=spec, start_step=0,
                    stop_step=int(root["remaining_steps"]), observation=observation,
                    official=official, dropped=dropped, rewards=rewards, trace=trace,
                    vision_actor=vision_actor, vision_stats=vision_stats,
                    oracle_actor=oracle_actor, oracle_stats=oracle_stats,
                    safe_policy=None, device=device, rho=rho,
                )
                metrics = _event_metrics(root, official, dropped, rewards, gamma, complete=finished)
                corrections = [x["correction_rms"] for x in trace]
                candidates.append({
                    "candidate_id": candidate_index, "candidate": _candidate_json(spec),
                    "descriptor": candidate_descriptor(spec).tolist(),
                    "outcome": metrics, "completion_mask": bool(finished),
                    "trace": trace,
                    "correction_rms": float(np.mean(corrections) if corrections else 0.0),
                })
                transitions += len(rewards)
            records.append({
                "root_id": root["root_id"], "source_episode_id": root["source_episode_seed"],
                "root_type": root["root_type"], "split": root["split"],
                "remaining_steps": root["remaining_steps"], "candidates": candidates,
                "branch_transitions": sum(x["outcome"]["steps"] for x in candidates),
            })
            if callback and (root_index + 1) % checkpoint_interval == 0:
                callback(records, transitions)
    finally:
        rollout_module._candidate_action = previous
        env.close()
    return records, transitions


def label_candidate(candidate: dict[str, Any], zero: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    c, z = candidate["outcome"], zero["outcome"]
    mask = bool(candidate.get("completion_mask") and zero.get("completion_mask"))
    if not mask:
        return {"mask": False, "strong_gain": 0, "harm": 0,
                "progress_delta": [0.0, 0.0, 0.0]}
    hold_requirement = 20
    strong = bool(
        (not z["strict"] and c["strict"])
        or (not z["benchmark"] and c["benchmark"])
        or (z["goal_exit_after_root"] and not c["goal_exit_after_root"])
        or (z["max_goal_streak"] < hold_requirement <= c["max_goal_streak"])
    )
    streak_drop = int(z["max_goal_streak"]) - int(c["max_goal_streak"])
    step_drop = int(z["goal_steps"]) - int(c["goal_steps"])
    harm = bool(
        (z["benchmark"] and not c["benchmark"])
        or (z["strict"] and not c["strict"])
        or (not z["goal_exit_after_root"] and c["goal_exit_after_root"])
        or streak_drop >= int(config["streak_significant_delta"])
        or step_drop >= int(config["goal_steps_significant_delta"])
    )
    z_exit = float(z["steps"] if not z["goal_exit_after_root"] else z["max_goal_streak"])
    c_exit = float(c["steps"] if not c["goal_exit_after_root"] else c["max_goal_streak"])
    return {"mask": True, "strong_gain": int(strong), "harm": int(harm),
            "progress_delta": [float(c["max_goal_streak"]-z["max_goal_streak"]),
                               float(c["goal_steps"]-z["goal_steps"]), c_exit-z_exit]}


def attach_labels(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    for root in records:
        zero = root["candidates"][0]
        for candidate in root["candidates"]:
            candidate["labels"] = label_candidate(candidate, zero, config)
    return records


def oracle_in_bank_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"roots": len(records), "by_root_type": {}}
    gains = 0; harms = 0; benchmark = 0; strict = 0; exits_avoided = 0
    zero_benchmark = 0; zero_strict = 0
    for root in records:
        zero = root["candidates"][0]
        valid = [c for c in root["candidates"] if c["labels"]["mask"]]
        best = max(valid, key=lambda c: (
            c["outcome"]["strict"], c["outcome"]["benchmark"],
            -c["outcome"]["goal_exit_after_root"], c["outcome"]["max_goal_streak"],
            c["outcome"]["goal_steps"], -c["correction_rms"],
            c["candidate"]["name"] == "zero"))
        gains += int(best["labels"]["strong_gain"])
        harms += int(best["labels"]["harm"])
        benchmark += int(best["outcome"]["benchmark"]); strict += int(best["outcome"]["strict"])
        zero_benchmark += int(zero["outcome"]["benchmark"]); zero_strict += int(zero["outcome"]["strict"])
        exits_avoided += int(zero["outcome"]["goal_exit_after_root"] and not best["outcome"]["goal_exit_after_root"])
    n=max(len(records),1)
    summary.update(zero_benchmark_rate=zero_benchmark/n, zero_strict_rate=zero_strict/n,
                   oracle_benchmark_rate=benchmark/n, oracle_strict_rate=strict/n,
                   roots_with_strong_gain=gains, strong_gain_rate=gains/n,
                   oracle_harms=harms, exits_avoided=exits_avoided)
    for phase in ROOT_PHASES:
        subset=[r for r in records if r["root_type"]==phase]
        summary["by_root_type"][phase]={
            "roots":len(subset),
            "roots_with_any_strong_gain":sum(any(c["labels"]["strong_gain"] for c in r["candidates"][1:]) for r in subset),
            "roots_with_any_harm":sum(any(c["labels"]["harm"] for c in r["candidates"][1:]) for r in subset),
        }
    return summary
