from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.counterfactual.branch_rollout import load_frozen_actor
from src.counterfactual.student_evaluation import evaluate_gated_student
from src.models.gated_residual_student import TemporalGatedVisualResidual
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a valid gated student")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "best_gated_student.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["gate_threshold"] is None:
        raise RuntimeError(
            "Checkpoint has no branch-dev-valid gate threshold and must not be evaluated "
            "as a closed-loop candidate"
        )
    seed_everything(
        int(config["seed"]), bool(config.get("deterministic_torch", True))
    )
    device = resolve_device(config["device"])
    base_actor, stats, _ = load_frozen_actor(checkpoint["base_checkpoint"], device)
    model = TemporalGatedVisualResidual(
        base_actor,
        **checkpoint["model_init"],
        gate_threshold=float(checkpoint["gate_threshold"]),
    ).to(device)
    model.load_state_dict(checkpoint["student_state"], strict=False)
    count = int(
        args.episodes
        if args.episodes is not None
        else config["validation_seeds"]["count"]
    )
    start = int(
        args.seed_start
        if args.seed_start is not None
        else config["validation_seeds"]["start"]
    )
    metrics = evaluate_gated_student(
        model,
        stats,
        range(start, start + count),
        device,
        float(checkpoint["gate_threshold"]),
    )
    output = (
        Path(args.output)
        if args.output
        else run_dir / "evaluations" / f"gated_{start}_{count}.json"
    )
    save_json(output, metrics)
    print(
        json.dumps(
            {"checkpoint": str(checkpoint_path), "output": str(output), **metrics},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
