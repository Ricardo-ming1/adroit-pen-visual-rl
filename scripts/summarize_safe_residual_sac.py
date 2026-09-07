from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the V3 Safe Residual pilot")
    parser.add_argument(
        "--run-dir",
        default="runs/v3/safe_residual_sac/anchor_0p05_seed_101",
    )
    parser.add_argument("--output-dir", default="results/v3")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    curve = pd.read_csv(run_dir / "validation_curve.csv")
    training = pd.read_csv(run_dir / "learning_curves" / "safe_residual_sac.csv")
    curve.to_csv(output / "safe_residual_validation_curve.csv", index=False)
    training.to_csv(output / "safe_residual_training_curve.csv", index=False)

    final_step = int(curve.online_steps.iloc[-1])
    baseline = _read_json(run_dir / "validation_step_000000.json")
    candidate = _read_json(run_dir / f"validation_step_{final_step:06d}.json")
    decision = _read_json(run_dir / f"safety_decision_{final_step:06d}.json")
    checkpoint = torch.load(
        run_dir / "best_safe_residual_validation.pt",
        map_location="cpu",
        weights_only=False,
    )

    summary_rows = []
    for method, step, metrics in (
        ("Frozen Vision AWAC E0", 0, baseline),
        ("Safe Residual SAC candidate", final_step, candidate),
    ):
        calibration = metrics.get("critic_calibration", {})
        summary_rows.append(
            {
                "method": method,
                "online_steps": step,
                "benchmark_success": metrics["benchmark_success"],
                "strict_stable_success": metrics["strict_stable_success"],
                "goal_entry_rate": metrics["goal_entry_rate"],
                "goal_exit_after_entry_rate": metrics["goal_exit_after_entry_rate"],
                "consecutive_success_window_rate": metrics[
                    "consecutive_success_window_rate"
                ],
                "correction_rms": metrics["correction_rms"],
                "correction_rms_max_dim": metrics["correction_rms_max_dim"],
                "z_extreme_rate": metrics["z_extreme_rate"],
                "base_boundary_rate": metrics["base_boundary_rate"],
                "executed_action_boundary_rate": metrics[
                    "executed_action_boundary_rate"
                ],
                "residual_added_boundary_rate": metrics[
                    "residual_added_boundary_rate"
                ],
                "numerical_clamp_count": metrics["numerical_clamp_count"],
                "q_base_mean": calibration.get("q_base_mean"),
                "mc_return_mean": calibration.get("mc_return_mean"),
                "q_bias_mean": calibration.get("q_bias_mean"),
                "q_mae": calibration.get("q_mae"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output / "safe_residual_paired_summary.csv", index=False
    )

    baseline_episodes = {episode["seed"]: episode for episode in baseline["episodes"]}
    candidate_episodes = {
        episode["seed"]: episode for episode in candidate["episodes"]
    }
    paired_rows = []
    for seed in sorted(baseline_episodes):
        base_episode = baseline_episodes[seed]
        safe_episode = candidate_episodes[seed]
        paired_rows.append(
            {
                "episode_seed": seed,
                "e0_benchmark": int(base_episode["benchmark_success"]),
                "safe_benchmark": int(safe_episode["benchmark_success"]),
                "benchmark_change": int(safe_episode["benchmark_success"])
                - int(base_episode["benchmark_success"]),
                "e0_strict": int(base_episode["strict_stable_success"]),
                "safe_strict": int(safe_episode["strict_stable_success"]),
                "strict_change": int(safe_episode["strict_stable_success"])
                - int(base_episode["strict_stable_success"]),
            }
        )
    pd.DataFrame(paired_rows).to_csv(
        output / "safe_residual_paired_episodes.csv", index=False
    )

    selected_steps = int(checkpoint["progress"]["online_steps"])
    status = {
        "development_seed": int(checkpoint["seed"]),
        "validation_episode_seeds": [
            int(min(baseline_episodes)),
            int(max(baseline_episodes)),
        ],
        "validation_episode_count": len(baseline_episodes),
        "pilot_final_steps": final_step,
        "pilot_passed": bool(decision["pass"]),
        "pilot_failures": decision["failures"],
        "anchor_rescue_allowed": bool(decision["anchor_rescue_allowed"]),
        "selected_checkpoint": str(
            run_dir / "best_safe_residual_validation.pt"
        ),
        "selected_online_steps": selected_steps,
        "candidate_checkpoint": str(run_dir / "last_safe_residual_sac.pt"),
        "candidate_online_steps": final_step,
        "continued_to_100k": False,
        "paired_confirmation_200_episodes": "not_run_after_30k_gate_failure",
        "three_seed_handoff": "not_run_after_30k_gate_failure",
        "hold_curriculum": "not_run_after_30k_gate_failure",
        "frozen_test": "not_run_without_a_promoted_candidate",
        "performance_improved": False,
    }
    _write_json(output / "experiment_status.json", status)
    pd.DataFrame(
        [
            {
                "experiment": "100k single-seed continuation",
                "status": "not_run",
                "reason": "30k final point failed benchmark, strict, and goal-entry floors",
            },
            {
                "experiment": "lambda_anchor=0.10 rescue",
                "status": "not_run",
                "reason": "rescue was allowed only for isolated correction/z failures",
            },
            {
                "experiment": "3-seed safe handoff",
                "status": "not_run",
                "reason": "single-seed 100k prerequisite was not reached",
            },
            {
                "experiment": "H0/H1 hold curriculum",
                "status": "not_run",
                "reason": "3-seed safe handoff prerequisite was not reached",
            },
        ]
    ).to_csv(output / "downstream_experiments.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    x = curve.online_steps / 1000
    axes[0, 0].plot(x, 100 * curve.benchmark_success, marker="o", label="benchmark")
    axes[0, 0].plot(x, 100 * curve.strict_stable_success, marker="o", label="strict")
    axes[0, 0].axhline(60, color="C0", linestyle="--", alpha=0.5)
    axes[0, 0].axhline(32, color="C1", linestyle="--", alpha=0.5)
    axes[0, 0].set_ylabel("Success (%)")
    axes[0, 0].legend()
    axes[0, 1].plot(x, curve.correction_rms, marker="o", label="all-dim RMS")
    axes[0, 1].plot(x, curve.correction_rms_max_dim, marker="o", label="max-dim RMS")
    axes[0, 1].axhline(0.04, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].set_ylabel("Executed correction")
    axes[0, 1].legend()
    axes[1, 0].plot(x, 100 * curve.residual_added_boundary_rate, marker="o")
    axes[1, 0].set_ylabel("Residual-added boundary (%)")
    axes[1, 1].plot(x, curve.q_base_mean, marker="o", label="Q(base)")
    axes[1, 1].plot(x, curve.mc_return_mean, marker="o", label="finite-horizon MC")
    axes[1, 1].plot(x, curve.q_bias_mean, marker="o", label="Q - MC bias")
    axes[1, 1].set_ylabel("Scaled return units")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Online transitions (thousands)")
        axis.grid(alpha=0.3)
    fig.suptitle("Safe Residual SAC: seed 101, 100 fixed validation episodes")
    fig.tight_layout()
    fig.savefig(output / "safe_residual_validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
