from __future__ import annotations

import argparse

from src.training import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Oracle Adroit Pen pipeline")
    parser.add_argument("--config", default="configs/oracle_full.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage", choices=["all", "bc", "critic", "awac", "ppo"], default="all"
    )
    parser.add_argument("--no-anchor", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = train_pipeline(
        args.config,
        "oracle",
        args.seed,
        args.stage,
        args.no_anchor,
        args.smoke,
        args.resume,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
