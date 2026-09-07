from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot frozen Vision validation curves")
    parser.add_argument("--run-root", default="runs/frozen/vision")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--output-dir", default="results/learning_curves")
    args = parser.parse_args()
    root, output = Path(args.run_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    stages = {
        "bc": ("bc.csv", "epoch", "BC epoch"),
        "awac": ("offline_awac.csv", "update", "AWAC update"),
        "ppo": ("online_ppo.csv", "online_steps", "Online environment steps"),
    }
    for stage, (filename, x_column, x_label) in stages.items():
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
        for seed in args.seeds:
            run_dir = root / f"full_seed_{seed}"
            curve = pd.read_csv(run_dir / "learning_curves" / filename)
            if stage == "ppo":
                initial = pd.read_csv(
                    run_dir / "learning_curves" / "online_ppo_validation.csv"
                ).iloc[[0]]
                curve = pd.concat([initial, curve], ignore_index=True)
            for axis, metric, title in zip(
                axes,
                ("benchmark_success", "strict_stable_success"),
                ("Benchmark success", "Strict stable success"),
            ):
                axis.plot(curve[x_column], curve[metric], marker="o", linewidth=1.2, label=f"seed {seed}")
                axis.set_title(title)
                axis.set_xlabel(x_label)
                axis.set_ylim(-0.03, 1.03)
                axis.grid(alpha=0.25)
        axes[0].set_ylabel("Validation success rate")
        axes[1].legend(fontsize=8, ncol=2)
        figure.suptitle(f"Frozen Vision {stage.upper()} validation")
        figure.tight_layout()
        figure.savefig(output / f"vision_{stage}_validation.png", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
