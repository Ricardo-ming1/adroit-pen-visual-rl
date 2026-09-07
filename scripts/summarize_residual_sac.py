from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def _aggregate(run_dir: Path, residual_scale: float) -> tuple[pd.DataFrame, dict]:
    curve = pd.read_csv(run_dir / "validation_curve.csv")
    curve.insert(0, "residual_scale", residual_scale)
    curve.insert(0, "run_dir", str(run_dir))
    checkpoint = torch.load(
        run_dir / "best_residual_sac_validation.pt",
        map_location="cpu",
        weights_only=False,
    )
    nonzero = curve[curve.online_steps > 0]
    ordered = nonzero.sort_values(
        ["benchmark_success", "strict_stable_success", "episode_return"],
        ascending=False,
    )
    best_nonzero = ordered.iloc[0]
    final = curve.iloc[-1]
    training = pd.read_csv(run_dir / "learning_curves" / "residual_sac.csv")
    diagnostic = training.iloc[-1]
    return curve, {
        "residual_scale": residual_scale,
        "trained_online_steps": int(final.online_steps),
        "selected_online_steps": int(checkpoint["progress"]["online_steps"]),
        "selected_benchmark": float(checkpoint["metrics"]["benchmark_success"]),
        "selected_strict": float(checkpoint["metrics"]["strict_stable_success"]),
        "best_nonzero_steps": int(best_nonzero.online_steps),
        "best_nonzero_benchmark": float(best_nonzero.benchmark_success),
        "best_nonzero_strict": float(best_nonzero.strict_stable_success),
        "final_benchmark": float(final.benchmark_success),
        "final_strict": float(final.strict_stable_success),
        "final_residual_l2": float(diagnostic.rollout_residual_l2_executed),
        "final_scaled_residual_l2": float(diagnostic.rollout_scaled_residual_l2_executed),
        "final_saturation_rate": float(
            diagnostic.rollout_combined_action_saturation_rate_executed
        ),
        "final_q_mean": float(diagnostic.update_q_mean),
        "final_q_target_mean": float(diagnostic.update_q_target_mean),
        "final_critic_loss": float(diagnostic.update_critic_loss),
        "final_temperature": float(diagnostic.update_temperature),
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best_residual_sac_validation.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the V2 Residual SAC development run")
    parser.add_argument("--main-root", default="runs/v2/residual_sac")
    parser.add_argument("--rescue-root", default="runs/v2/residual_sac_rescue")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-dir", default="results/v2")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = [
        (Path(args.main_root) / f"scale_0p2_seed_{args.seed}", 0.2),
        (Path(args.rescue_root) / f"scale_0p1_seed_{args.seed}", 0.1),
        (Path(args.rescue_root) / f"scale_0p4_seed_{args.seed}", 0.4),
    ]
    curves = []
    summaries = []
    for run_dir, scale in runs:
        curve, summary = _aggregate(run_dir, scale)
        curves.append(curve)
        summaries.append(summary)
    combined = pd.concat(curves, ignore_index=True)
    combined.to_csv(output / "residual_sac_validation_curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "residual_sac_summary.csv", index=False)

    baseline = combined[(combined.residual_scale == 0.2) & (combined.online_steps == 0)].iloc[0]
    chosen = next(row for row in summaries if row["residual_scale"] == 0.1)
    status = {
        "development_seed": args.seed,
        "frozen_awac": {
            "benchmark_success": float(baseline.benchmark_success),
            "strict_stable_success": float(baseline.strict_stable_success),
        },
        "rescue_selected_by_validation": "residual_scale=0.1",
        "residual_sac_promotion_passed": False,
        "promotion_reasons": [
            "no nonzero checkpoint met the E0 strict-minus-2pp selection floor",
            "best nonzero checkpoint did not improve benchmark by 5 percentage points",
            "default scale 0.2 showed persistent action saturation and Q-scale drift",
        ],
        "selected_checkpoint": chosen["best_checkpoint"],
        "selected_online_steps": chosen["selected_online_steps"],
        "paired_three_seed_status": "not_run_after_development_failure",
        "hold_curriculum_status": "not_run_after_residual_sac_failed_promotion",
        "frozen_five_seed_test_status": "not_run_without_a_promoted_candidate",
    }
    with (output / "experiment_status.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
        handle.write("\n")
    pd.DataFrame(
        [
            {
                "method": "Residual SAC E1 paired validation",
                "status": "not_run",
                "reason": "development seed failed promotion after both allowed rescue pilots",
            },
            {
                "method": "H0 normal continuation",
                "status": "not_run",
                "reason": "Residual SAC did not pass the prerequisite",
            },
            {
                "method": "H1 hold curriculum",
                "status": "not_run",
                "reason": "Residual SAC did not pass the prerequisite",
            },
            {
                "method": "5-seed frozen test",
                "status": "not_run",
                "reason": "no promoted candidate was frozen",
            },
        ]
    ).to_csv(output / "downstream_experiments.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for scale in (0.1, 0.2, 0.4):
        curve = combined[combined.residual_scale == scale]
        label = f"residual_scale={scale}"
        axes[0].plot(curve.online_steps, 100 * curve.benchmark_success, marker="o", label=label)
        axes[1].plot(curve.online_steps, 100 * curve.strict_stable_success, marker="o", label=label)
    axes[0].set_title("Benchmark success")
    axes[1].set_title("Strict stable success")
    for axis in axes:
        axis.set_xlabel("Online environment transitions")
        axis.set_ylabel("Validation success (%)")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "residual_sac_validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
