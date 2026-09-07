from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.evaluation import wilson_interval


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(method: str, records: list[tuple[int, dict]]) -> tuple[dict, list[dict]]:
    per_seed = []
    all_episodes = []
    for seed, metrics in records:
        episodes = metrics["episodes"]
        all_episodes.extend(episodes)
        per_seed.append(
            {
                "method": method,
                "training_seed": seed,
                "episodes": len(episodes),
                **{key: value for key, value in metrics.items() if key != "episodes"},
            }
        )

    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in per_seed], dtype=np.float64)

    def pooled(key: str) -> tuple[float, float, float]:
        successes = sum(bool(episode[key]) for episode in all_episodes)
        low, high = wilson_interval(successes, len(all_episodes))
        return successes / len(all_episodes), low, high

    benchmark_pooled, benchmark_low, benchmark_high = pooled("benchmark_success")
    strict_pooled, strict_low, strict_high = pooled("strict_stable_success")
    ddof = 1 if len(records) > 1 else 0
    summary = {
        "method": method,
        "training_seeds": len(records),
        "total_episodes": len(all_episodes),
        "benchmark_mean": values("benchmark_success").mean(),
        "benchmark_std": values("benchmark_success").std(ddof=ddof),
        "benchmark_pooled": benchmark_pooled,
        "benchmark_wilson_low": benchmark_low,
        "benchmark_wilson_high": benchmark_high,
        "benchmark_worst_seed": values("benchmark_success").min(),
        "strict_mean": values("strict_stable_success").mean(),
        "strict_std": values("strict_stable_success").std(ddof=ddof),
        "strict_pooled": strict_pooled,
        "strict_wilson_low": strict_low,
        "strict_wilson_high": strict_high,
        "strict_worst_seed": values("strict_stable_success").min(),
        "drop_rate_mean": values("drop_rate").mean(),
        "final_orientation_error_mean": values("final_orientation_error").mean(),
        "episode_return_mean": values("episode_return").mean(),
        "action_magnitude_mean": values("action_magnitude").mean(),
        "action_smoothness_mean": values("action_smoothness").mean(),
        "non_finite_episodes": int(sum(row["non_finite_episodes"] for row in per_seed)),
    }
    times = [row["time_to_success"] for row in per_seed if row["time_to_success"] is not None]
    summary["time_to_success_mean"] = float(np.mean(times)) if times else None
    return summary, per_seed


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate held-out Adroit Pen results")
    parser.add_argument("--run-root", default="runs/frozen/vision")
    parser.add_argument("--full-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--baseline-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    root, output = Path(args.run_root), Path(args.output_dir)
    specifications = [
        ("Vision BC", "full", "bc", args.baseline_seeds),
        ("Vision BC + AWAC", "full", "awac", args.baseline_seeds),
        ("Vision BC + AWAC + PPO/KL/BC", "full", "ppo", args.full_seeds),
        ("Full method without KL/BC anchor", "full_no_anchor", "ppo", args.baseline_seeds),
    ]
    summaries, seed_rows = [], []
    for method, variant, stage, seeds in specifications:
        records = []
        for seed in seeds:
            path = root / f"{variant}_seed_{seed}" / "evaluations" / f"{stage}_test_normal.json"
            records.append((seed, load(path)))
        summary, per_seed = summarize(method, records)
        summaries.append(summary)
        seed_rows.extend(per_seed)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "final_test_by_seed.csv", seed_rows)

    interventions = []
    for intervention in ("normal", "black", "episode_shuffle", "temporal_shuffle", "target_occlusion"):
        records = []
        for seed in args.full_seeds:
            path = root / f"full_seed_{seed}" / "evaluations" / f"ppo_test_{intervention}.json"
            records.append((seed, load(path)))
        summary, _ = summarize(intervention, records)
        interventions.append(summary)
    write_csv(output / "visual_interventions.csv", interventions)
    print(output / "summary.csv")


if __name__ == "__main__":
    main()
