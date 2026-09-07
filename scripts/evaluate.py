from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.offline import NormalizationStats
from src.evaluation import evaluate_policy
from src.models.actor import SquashedGaussianActor
from src.utils import load_yaml, resolve_device, save_json, seed_everything


CHECKPOINTS = {
    "bc": "best_bc_validation.pt",
    "awac": "best_offline_awac_validation.pt",
    "ppo": "best_online_ppo_validation.pt",
}
INTERVENTIONS = (
    "normal",
    "black",
    "episode_shuffle",
    "temporal_shuffle",
    "target_occlusion",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Adroit Pen checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=sorted(CHECKPOINTS), default="ppo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--intervention", choices=INTERVENTIONS, default="normal")
    parser.add_argument("--output")
    parser.add_argument("--video-dir")
    parser.add_argument("--video-episodes", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / CHECKPOINTS[args.stage]
    device = resolve_device(config.get("device", "cuda"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    seed_everything(int(checkpoint["seed"]), bool(config.get("deterministic_torch", True)))

    actor = SquashedGaussianActor(**checkpoint["actor_init"]).to(device)
    actor.load_state_dict(checkpoint["actor_state"])
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    seed_config = config[f"{args.split}_seeds"]
    count = int(args.episodes or seed_config["count"])
    start = int(args.seed_start if args.seed_start is not None else seed_config["start"])
    metrics = evaluate_policy(
        actor,
        checkpoint["line"],
        stats,
        range(start, start + count),
        device,
        intervention=args.intervention,
        video_dir=args.video_dir,
        video_episodes=int(args.video_episodes),
    )
    output = (
        Path(args.output)
        if args.output
        else run_dir / "evaluations" / f"{args.stage}_{args.split}_{args.intervention}.json"
    )
    save_json(output, metrics)
    aggregate = {key: value for key, value in metrics.items() if key != "episodes"}
    print(json.dumps({"checkpoint": str(checkpoint_path), "output": str(output), **aggregate}, indent=2))


if __name__ == "__main__":
    main()
