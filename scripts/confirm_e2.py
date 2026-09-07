from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_state_distilled_policy import load_estimator
from src.counterfactual.branch_rollout import deterministic_action, load_frozen_actor
from src.envs.adroit import HORIZON, VisualAdroitEnv
from src.state_distillation.evaluation import DistilledOraclePolicy
from src.state_distillation.v6_1 import aggregate, episode_metrics, paired_binary_statistics, sha256_file
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def run_episode(env, seed, *, line, actor, stats, distilled=None):
    observation, _ = env.reset(seed=int(seed))
    if distilled is not None:
        distilled.reset()
    official: list[bool] = []
    dropped: list[bool] = []
    finite = True
    for _ in range(HORIZON):
        if distilled is None:
            action = deterministic_action(actor, line, observation, stats, actor_device(actor))
        else:
            action, _ = distilled.act(observation)
        finite = finite and bool(np.isfinite(action).all())
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        observation, _, terminated, truncated, info = env.step(action)
        official.append(bool(info["official_goal"]))
        dropped.append(bool(info["dropped"]))
        if terminated or truncated:
            break
    return episode_metrics(int(seed), official, dropped, finite)


def actor_device(actor) -> torch.device:
    return next(actor.parameters()).device


def resumable_policy_eval(name, output, seeds, chunk_size, *, actor, stats, line, distilled=None):
    output = Path(output)
    episodes: list[dict] = []
    if output.exists():
        prior = json.loads(output.read_text())
        episodes = list(prior.get("episodes", []))
    done = {int(row["seed"]) for row in episodes}
    env = VisualAdroitEnv()
    try:
        for index, seed in enumerate(seeds):
            if int(seed) in done:
                continue
            episodes.append(run_episode(env, int(seed), line=line, actor=actor, stats=stats, distilled=distilled))
            if len(episodes) % int(chunk_size) == 0:
                save_json(output, {"policy": name, **aggregate(sorted(episodes, key=lambda x: x["seed"]))})
                print(json.dumps({"policy": name, "completed": len(episodes)}), flush=True)
    finally:
        env.close()
    result = {"policy": name, **aggregate(sorted(episodes, key=lambda x: x["seed"]))}
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v6_1_confirmation.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    run_dir, results_dir = Path(config["run_dir"]), Path(config["results_dir"])
    run_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True)
    if sha256_file(config["e2_checkpoint"]) != config["e2_checkpoint_sha256"]:
        raise RuntimeError("frozen E2 checkpoint SHA-256 mismatch")
    v6 = json.loads(Path(config["v6_summary"]).read_text())
    if v6.get("final_status") != config["v6_required_status"]:
        raise RuntimeError("V6 original terminal status is not preserved")
    phase = config["phase_a"]
    seeds = list(range(int(phase["seed_start"]), int(phase["seed_start"]) + int(phase["episodes"])))
    seed_everything(61101, True)
    device = resolve_device(config["device"])
    vision, vision_stats, _ = load_frozen_actor(config["vision_checkpoint"], device)
    oracle, oracle_stats, _ = load_frozen_actor(config["oracle_checkpoint"], device)
    estimator, _ = load_estimator(config["e2_checkpoint"], vision, device)
    e2 = DistilledOraclePolicy(estimator, oracle, oracle_stats, vision_stats, device)
    baseline = resumable_policy_eval("Frozen Vision AWAC", run_dir / "phase_a_e0.partial.json", seeds,
        phase["chunk_size"], actor=vision, stats=vision_stats, line="vision")
    candidate = resumable_policy_eval("Frozen E2", run_dir / "phase_a_e2.partial.json", seeds,
        phase["chunk_size"], actor=oracle, stats=oracle_stats, line="oracle", distilled=e2)
    reference = resumable_policy_eval("True-state Frozen Oracle AWAC", run_dir / "phase_a_oracle.partial.json", seeds,
        phase["chunk_size"], actor=oracle, stats=oracle_stats, line="oracle")
    paired = {}
    for key, label in (("benchmark_success", "benchmark"), ("strict_stable_success", "strict")):
        paired[label] = paired_binary_statistics(candidate["episodes"], baseline["episodes"], key,
            bootstrap_replicates=int(phase["bootstrap_replicates"]),
            bootstrap_seed=int(phase["bootstrap_seed"]) + (0 if label == "benchmark" else 1))
    bdelta = candidate["benchmark_success"] - baseline["benchmark_success"]
    sdelta = candidate["strict_stable_success"] - baseline["strict_stable_success"]
    acquisition = bdelta >= float(phase["acquisition_benchmark_delta"]) and sdelta >= float(phase["acquisition_strict_floor"])
    strict = sdelta >= float(phase["strict_delta"]) and bdelta >= float(phase["strict_benchmark_floor"])
    result = {
        "v6_formal_status_preserved": v6["final_status"],
        "e2_checkpoint": str(config["e2_checkpoint"]),
        "e2_checkpoint_sha256": sha256_file(config["e2_checkpoint"]),
        "seed_start": seeds[0], "seed_end": seeds[-1], "episodes": len(seeds),
        "policies": {"e0": {k:v for k,v in baseline.items() if k != "episodes"},
                     "e2": {k:v for k,v in candidate.items() if k != "episodes"},
                     "oracle": {k:v for k,v in reference.items() if k != "episodes"}},
        "paired_e2_vs_e0": paired,
        "benchmark_difference": bdelta, "strict_difference": sdelta,
        "acquisition_criterion_pass": acquisition, "strict_criterion_pass": strict,
        "phase_a_confirmation_pass": bool(acquisition or strict),
        "privileged_state_read_by_e2_policy": False,
        "vision_warmup_used": False, "controller_switching_used": False,
    }
    save_json(results_dir / "phase_a_confirmation300.json", result)
    save_json(run_dir / "phase_a_complete.json", {"complete": True, "phase_a_confirmation_pass": result["phase_a_confirmation_pass"]})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
