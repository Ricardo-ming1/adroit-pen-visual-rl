from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import evaluate_policy
from src.residual_training import load_residual_policy
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Residual SAC checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--output")
    parser.add_argument("--video-dir")
    parser.add_argument("--video-episodes", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best_residual_sac_validation.pt"
    device = resolve_device(config.get("device", "cuda"))
    policy, stats, checkpoint = load_residual_policy(checkpoint_path, device)
    seed_everything(int(checkpoint["seed"]), bool(config.get("deterministic_torch", True)))
    bank = config[f"{args.split}_seeds"]
    count = int(args.episodes if args.episodes is not None else bank["count"])
    start = int(args.seed_start if args.seed_start is not None else bank["start"])
    metrics = evaluate_policy(
        policy,
        "vision",
        stats,
        range(start, start + count),
        device,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
    )
    output = Path(args.output) if args.output else run_dir / "evaluations" / f"residual_{args.split}.json"
    save_json(output, metrics)
    aggregate = {key: value for key, value in metrics.items() if key != "episodes"}
    print(json.dumps({"checkpoint": str(checkpoint_path), "selected_online_steps": checkpoint["progress"]["online_steps"], "output": str(output), **aggregate}, indent=2))


if __name__ == "__main__":
    main()
