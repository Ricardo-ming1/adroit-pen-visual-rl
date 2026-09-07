from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.counterfactual.branch_rollout import deterministic_action, load_frozen_actor
from src.envs.adroit import HORIZON, VisualAdroitEnv
from src.state_distillation.v6_1 import CausalHistory
from src.utils import load_yaml, resolve_device, save_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/v6_1_confirmation.yaml")
    args = parser.parse_args(); config = load_yaml(args.config); spec = config["fixed_entry_hold"]
    device = resolve_device(config["device"])
    vision, vision_stats, _ = load_frozen_actor(config["vision_checkpoint"], device)
    output = Path(config["run_dir"]) / "fixed_entry_roots.pt"; output.parent.mkdir(parents=True, exist_ok=True)
    roots: list[dict] = []
    env = VisualAdroitEnv(); attempted = 0
    try:
        for seed in range(int(spec["seed_start"]), int(spec["seed_start"]) + int(spec["maximum_source_episodes"])):
            if len(roots) >= int(spec["roots"]): break
            attempted += 1
            observation, _ = env.reset(seed=seed); history = CausalHistory(); history.reset(observation); previously_inside = False
            for t in range(HORIZON):
                action = deterministic_action(vision, "vision", observation, vision_stats, device)
                observation, _, terminated, truncated, info = env.step(action); history.append(observation)
                entered = bool(info["official_goal"] and not previously_inside)
                remaining = HORIZON - (t + 1)
                if entered and remaining >= int(spec["minimum_remaining_steps"]):
                    roots.append({"root_id": len(roots), "source_seed": seed, "entry_step": t + 1,
                        "remaining_horizon": remaining, "training_state": env.training_state(),
                        "history_before_current": history.before_current()})
                    if len(roots) % 20 == 0: print(json.dumps({"roots": len(roots), "source_episodes": attempted}), flush=True)
                    break
                previously_inside = bool(info["official_goal"])
                if terminated or truncated: break
    finally: env.close()
    if len(roots) != int(spec["roots"]):
        raise RuntimeError(f"only collected {len(roots)} of {spec['roots']} fixed-entry roots")
    torch.save({"schema_version": 1, "selection": "E0 first goal entry; no future-exit filtering",
                "source_seed_start": int(spec["seed_start"]), "source_episodes_attempted": attempted,
                "roots": roots}, output)
    save_json(Path(config["results_dir"]) / "fixed_entry_bank_summary.json",
        {"roots": len(roots), "source_episodes_attempted": attempted, "source_seed_start": int(spec["seed_start"]),
         "source_seed_end": int(spec["seed_start"]) + attempted - 1, "future_exit_filter_used": False,
         "minimum_remaining_steps": int(spec["minimum_remaining_steps"]), "bank_path": str(output)})
    print(json.dumps({"roots": len(roots), "source_episodes_attempted": attempted, "output": str(output)}, indent=2))


if __name__ == "__main__": main()
