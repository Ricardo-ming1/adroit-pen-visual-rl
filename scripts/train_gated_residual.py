from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.counterfactual.branch_dataset import load_root_dataset
from src.counterfactual.branch_rollout import load_frozen_actor
from src.counterfactual.student_evaluation import evaluate_gated_student
from src.counterfactual.student_training import train_gated_student
from src.data.offline import PenOfflineData
from src.evaluation import evaluate_policy
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_paired(path: Path, baseline: dict, student: dict) -> None:
    baseline_by_seed = {episode["seed"]: episode for episode in baseline["episodes"]}
    student_by_seed = {episode["seed"]: episode for episode in student["episodes"]}
    rows = []
    for seed in sorted(baseline_by_seed):
        first = baseline_by_seed[seed]
        second = student_by_seed[seed]
        rows.append(
            {
                "episode_seed": seed,
                "e0_benchmark": int(first["benchmark_success"]),
                "student_benchmark": int(second["benchmark_success"]),
                "benchmark_change": int(second["benchmark_success"])
                - int(first["benchmark_success"]),
                "e0_strict": int(first["strict_stable_success"]),
                "student_strict": int(second["strict_stable_success"]),
                "strict_change": int(second["strict_stable_success"])
                - int(first["strict_stable_success"]),
                "e0_goal_exit": int(first["goal_exit_after_entry"]),
                "student_goal_exit": int(second["goal_exit_after_entry"]),
                "student_gate_activations": second["gate_activations"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Round-0 gated student")
    parser.add_argument("--config", default="configs/counterfactual_residual.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(
        int(config["seed"]), bool(config.get("deterministic_torch", True))
    )
    device = resolve_device(config["device"])
    run_dir = Path(config["run_dir"])
    data_config = load_yaml(config["data_config"])
    data = PenOfflineData(
        data_config["output_path"], frame_stack=int(data_config["frame_stack"])
    )
    roots, _ = load_root_dataset(run_dir / "branch_roots.pt")
    branch_results = _load_json(run_dir / "branch_labels.json")["results"]
    labels = {result["root_id"] for result in branch_results}
    if labels != {root["root_id"] for root in roots}:
        raise ValueError("Branch labels and roots differ")
    train_roots = [root for root in roots if root["split"] == "train"]
    dev_roots = [root for root in roots if root["split"] == "dev"]
    base_actor, stats, _ = load_frozen_actor(config["vision_checkpoint"], device)
    model, training_summary = train_gated_student(
        base_actor=base_actor,
        base_checkpoint_path=config["vision_checkpoint"],
        stats=stats,
        train_roots=train_roots,
        dev_roots=dev_roots,
        branch_results=branch_results,
        config=config,
        device=device,
        run_dir=run_dir,
    )
    save_json(run_dir / "student_branch_dev_summary.json", training_summary)
    if not training_summary["gate_dev_pass"]:
        status = {
            "round0_status": "not_run_gate_dev_failed",
            "visual_rescue_eligible": bool(
                training_summary["residual_fit_pass"]
            ),
            **training_summary,
        }
        save_json(run_dir / "student_status.json", status)
        print(json.dumps(status, indent=2))
        return

    threshold = float(training_summary["selected_threshold"]["threshold"])
    model.gate_threshold = threshold
    seeds = range(
        int(config["validation_seeds"]["start"]),
        int(config["validation_seeds"]["start"])
        + int(config["validation_seeds"]["count"]),
    )
    baseline = evaluate_policy(base_actor, "vision", data.stats, seeds, device)
    student = evaluate_gated_student(model, data.stats, seeds, device, threshold)
    save_json(run_dir / "round0_e0_validation.json", baseline)
    save_json(run_dir / "round0_student_validation.json", student)
    _write_paired(run_dir / "round0_paired_episodes.csv", baseline, student)
    round0_pass = bool(
        student["benchmark_success"] >= baseline["benchmark_success"] - 0.02
        and student["strict_stable_success"]
        >= baseline["strict_stable_success"] - 0.02
        and student["drop_rate"] <= baseline["drop_rate"] + 0.02
        and student["gate_activation_rate"] > 0.0
        and student["goal_exit_after_entry_rate"]
        <= baseline["goal_exit_after_entry_rate"] + 0.02
    )
    status = {
        "round0_status": "pass" if round0_pass else "failed_closed_loop_safety",
        "round0_pass": round0_pass,
        "gate_threshold": threshold,
        "branch_dev_gate_metrics": training_summary["selected_threshold"],
        "e0": {
            key: baseline[key]
            for key in (
                "benchmark_success",
                "strict_stable_success",
                "goal_entry_rate",
                "goal_exit_after_entry_rate",
                "drop_rate",
            )
        },
        "student": {
            key: student[key]
            for key in (
                "benchmark_success",
                "strict_stable_success",
                "goal_entry_rate",
                "goal_exit_after_entry_rate",
                "drop_rate",
                "gate_activation_rate",
                "episodes_with_gate",
                "correction_rms",
                "numerical_clamp_count",
            )
        },
        "dagger_round1_status": "eligible" if round0_pass else "not_run",
    }
    save_json(run_dir / "student_status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
