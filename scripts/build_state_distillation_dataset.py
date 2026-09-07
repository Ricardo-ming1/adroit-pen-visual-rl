from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from src.counterfactual.branch_rollout import deterministic_action, load_frozen_actor
from src.envs.adroit import HORIZON, VisualAdroitEnv, current_metrics, privileged_observation
from src.state_distillation.dataset import append_episode, episode_auxiliary_labels
from src.state_distillation.oracle_observation import OracleObservationAdapter
from src.utils import load_yaml, resolve_device, seed_everything


def existing(path: Path) -> set[int]:
    if not path.exists(): return set()
    with h5py.File(path, "r") as handle:
        return {int(group.attrs["seed"]) for group in handle["episodes"].values()}


@torch.inference_mode()
def collect_episode(env, seed, source, execution_actor, execution_line, execution_stats,
                    oracle_actor, oracle_stats, device):
    observation, _ = env.reset(seed=int(seed)); fields: dict[str, list] = {
        name: [] for name in ("rgb", "proprio", "previous_action", "oracle_observation",
                              "primitive", "oracle_action", "executed_action", "official")}
    adapter = OracleObservationAdapter()
    for _ in range(HORIZON):
        oracle_observation = privileged_observation(env.env)
        teacher = deterministic_action(oracle_actor, "oracle", observation, oracle_stats, device)
        action = deterministic_action(execution_actor, execution_line, observation, execution_stats, device)
        fields["rgb"].append(observation["rgb"][-1].transpose(1, 2, 0).copy())
        fields["proprio"].append(observation["proprio"].copy())
        fields["previous_action"].append(observation["previous_action"].copy())
        fields["oracle_observation"].append(oracle_observation)
        fields["primitive"].append(adapter.primitive_from_runtime(env.env))
        fields["oracle_action"].append(teacher)
        fields["executed_action"].append(action)
        fields["official"].append(current_metrics(env.env)["official_goal"])
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated: break
    arrays = {key: np.asarray(value) for key, value in fields.items() if key != "official"}
    phase, exit_10, exit_20 = episode_auxiliary_labels(np.asarray(fields["official"], bool))
    arrays.update(phase=phase, exit_10=exit_10, exit_20=exit_20)
    return arrays


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/temporal_state_distillation.yaml")
    parser.add_argument("--output", default="runs/v6/source_sequences.h5"); parser.add_argument("--limit", type=int)
    args = parser.parse_args(); config = load_yaml(args.config); seed_everything(int(config["seed"]), True)
    device = resolve_device(config["device"]); vision, vision_stats, _ = load_frozen_actor(config["vision_checkpoint"], device)
    oracle, oracle_stats, _ = load_frozen_actor(config["oracle_checkpoint"], device)
    output = Path(args.output); done = existing(output); env = VisualAdroitEnv(); collected = 0
    plans = [
        ("v4_vision_source", int(config["source_data"]["v4_seed_start"]), int(config["source_data"]["v4_episodes"]), vision, "vision", vision_stats),
        ("oracle_training", int(config["source_data"]["oracle_seed_start"]), int(config["source_data"]["oracle_episodes"]), oracle, "oracle", oracle_stats),
    ]
    try:
        episode_id = 0
        if output.exists():
            with h5py.File(output, "r") as handle: episode_id = len(handle["episodes"])
        for source, start, count, actor, line, stats in plans:
            for seed in range(start, start + count):
                if seed in done: continue
                arrays = collect_episode(env, seed, source, actor, line, stats, oracle, oracle_stats, device)
                append_episode(output, episode_id, seed=seed, source=source, arrays=arrays)
                episode_id += 1; collected += 1
                if collected % 10 == 0: print(json.dumps({"collected": collected, "seed": seed, "source": source}), flush=True)
                if args.limit is not None and collected >= args.limit: raise StopIteration
    except StopIteration: pass
    finally: env.close()
    with h5py.File(output, "r") as handle:
        summary = {"episodes": len(handle["episodes"]), "transitions": sum(len(x["rgb"]) for x in handle["episodes"].values()),
                   "train_episodes": sum(str(x.attrs["split"]) == "train" for x in handle["episodes"].values()),
                   "dev_episodes": sum(str(x.attrs["split"]) == "dev" for x in handle["episodes"].values())}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
