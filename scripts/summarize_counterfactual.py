from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V4 counterfactual refinement")
    parser.add_argument("--run-dir", default="runs/v4/counterfactual_pilot")
    parser.add_argument("--output-dir", default="results/v4")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    root_summary = _read(run_dir / "root_collection_summary.json")
    zero = _read(run_dir / "zero_branch_validation.json")
    pilot = _read(run_dir / "branch_pilot_summary.json")
    labels = _read(run_dir / "branch_labels.json")["results"]
    student = _read(run_dir / "student_branch_dev_summary.json")
    student_status = _read(run_dir / "student_status.json")
    oracle = _read(Path("runs/v4/baselines/oracle_awac_validation100.json"))
    vision = _read(
        Path(
            "runs/v3/safe_residual_sac/anchor_0p05_seed_101/validation_step_000000.json"
        )
    )

    pd.read_csv(run_dir / "root_index.csv").to_csv(
        output / "root_index.csv", index=False
    )
    pd.read_csv(run_dir / "student_learning_curve.csv").to_csv(
        output / "student_learning_curve.csv", index=False
    )
    shutil.copy2(
        run_dir / "best_gated_student.pt",
        output / "gated_student_branch_dev_failed.pt",
    )
    _write(output / "zero_branch_validation.json", zero)

    root_rows = []
    for root_type, values in pilot["by_root_type"].items():
        root_rows.append({"root_type": root_type, **values})
    pd.DataFrame(root_rows).to_csv(
        output / "branch_positive_by_root_type.csv", index=False
    )

    positive = [result for result in labels if result["gate_target"]]
    winner_counts = Counter(result["winner"] for result in positive)
    source_counts = Counter(result["winner_source"] for result in positive)
    corrections = defaultdict(list)
    for result in positive:
        corrections[result["winner"]].append(result["winner_correction_rms"])
    winner_rows = [
        {
            "candidate": name,
            "positive_wins": count,
            "mean_correction_rms": float(np.mean(corrections[name])),
        }
        for name, count in winner_counts.most_common()
    ]
    pd.DataFrame(winner_rows).to_csv(
        output / "branch_winner_candidates.csv", index=False
    )

    event_improvements = Counter()
    positive_rows = []
    for result in positive:
        zero_full = result["zero"]["full"]
        winner = result["completed_candidates"][result["winner"]]["full"]
        improvements = {
            "strict": winner["strict"] > zero_full["strict"],
            "benchmark": winner["benchmark"] > zero_full["benchmark"],
            "goal_exit_avoided": winner["goal_exit_after_root"]
            < zero_full["goal_exit_after_root"],
            "max_goal_streak": winner["max_goal_streak"]
            > zero_full["max_goal_streak"],
            "goal_steps": winner["goal_steps"] > zero_full["goal_steps"],
        }
        event_improvements.update(
            key for key, improved in improvements.items() if improved
        )
        positive_rows.append(
            {
                "root_id": result["root_id"],
                "root_type": result["root_type"],
                "split": result["split"],
                "winner": result["winner"],
                "winner_source": result["winner_source"],
                "correction_rms": result["winner_correction_rms"],
                **{key: int(value) for key, value in improvements.items()},
            }
        )
    pd.DataFrame(positive_rows).to_csv(
        output / "branch_positive_roots.csv", index=False
    )

    threshold_frame = pd.DataFrame(student["threshold_rows"])
    threshold_frame.to_csv(output / "gate_thresholds.csv", index=False)
    baselines = pd.DataFrame(
        [
            {
                "method": "Frozen Vision AWAC",
                "episodes": 100,
                "benchmark_success": vision["benchmark_success"],
                "strict_stable_success": vision["strict_stable_success"],
                "goal_entry_rate": vision["goal_entry_rate"],
                "goal_exit_after_entry_rate": vision["goal_exit_after_entry_rate"],
            },
            {
                "method": "Oracle AWAC",
                "episodes": 100,
                "benchmark_success": oracle["benchmark_success"],
                "strict_stable_success": oracle["strict_stable_success"],
                "goal_entry_rate": oracle["goal_entry_rate"],
                "goal_exit_after_entry_rate": oracle[
                    "goal_exit_after_entry_rate"
                ],
            },
        ]
    )
    baselines.to_csv(output / "same_bank_baselines.csv", index=False)

    combined = {
        "phase": "counterfactual simulation-guided visual policy refinement",
        "roots": root_summary,
        "branch_pilot": pilot,
        "event_improvement_counts": dict(event_improvements),
        "positive_winner_sources": dict(source_counts),
        "positive_correction_rms_mean": float(
            np.mean([result["winner_correction_rms"] for result in positive])
        ),
        "positive_correction_rms_max": float(
            np.max([result["winner_correction_rms"] for result in positive])
        ),
        "student_branch_dev": {
            key: student[key]
            for key in (
                "train_roots",
                "dev_roots",
                "train_positive_roots",
                "dev_positive_roots",
                "best_dev_total_loss",
                "train_positive_residual_rmse",
                "train_zero_predictor_rmse",
                "train_rmse_relative_to_zero",
                "dev_positive_residual_rmse",
                "dev_zero_predictor_rmse",
                "dev_rmse_relative_to_zero",
                "residual_fit_pass",
                "selected_threshold",
                "gate_dev_pass",
            )
        },
    }
    _write(output / "summary.json", combined)

    status = {
        "branch_pilot_status": "passed_local_improvability",
        "student_engineering_status": "trained",
        "student_gate_dev_status": "failed",
        "visual_last_block_rescue_status": "not_eligible_residual_regression_did_not_fit",
        "round0_closed_loop": "not_run_without_valid_branch_dev_threshold",
        "dagger_round1": "not_run_round0_prerequisite_failed",
        "dagger_round2": "not_run_round0_prerequisite_failed",
        "paired_confirmation_200": "not_run_without_dagger_candidate",
        "three_seed": "not_run_without_confirmed_candidate",
        "frozen_test": "not_run_without_three_seed_promotion",
        "selected_deployable_checkpoint": "runs/frozen/vision/full_seed_101/best_offline_awac_validation.pt",
        "failed_student_checkpoint": "results/v4/gated_student_branch_dev_failed.pt",
        "student_status": student_status["round0_status"],
    }
    _write(output / "experiment_status.json", status)
    pd.DataFrame(
        [
            {"stage": key, "status": value}
            for key, value in status.items()
            if key not in ("selected_deployable_checkpoint", "failed_student_checkpoint")
        ]
    ).to_csv(output / "downstream_status.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    root_frame = pd.DataFrame(root_rows)
    axes[0].bar(
        root_frame.root_type.str.replace("_", "\n"),
        100 * root_frame.positive_root_rate,
    )
    axes[0].axhline(15, color="black", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Positive roots (%)")
    axes[0].set_title("True branch improvements")
    axes[1].plot(
        threshold_frame.threshold,
        100 * threshold_frame.precision,
        marker="o",
        label="precision",
    )
    axes[1].plot(
        threshold_frame.threshold,
        100 * threshold_frame.false_positive_rate,
        marker="o",
        label="false-positive rate",
    )
    axes[1].plot(
        threshold_frame.threshold,
        100 * threshold_frame.coverage,
        marker="o",
        label="coverage",
    )
    axes[1].axhline(80, color="C0", linestyle="--", alpha=0.4)
    axes[1].axhline(5, color="C1", linestyle="--", alpha=0.4)
    axes[1].set_xlabel("Gate threshold")
    axes[1].set_ylabel("Branch-dev metric (%)")
    axes[1].set_title("No threshold meets the gate requirements")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "counterfactual_pilot_and_gate.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
