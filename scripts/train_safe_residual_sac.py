from __future__ import annotations

import argparse

from src.safe_residual_training import train_safe_residual_sac


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Safe Frozen-AWAC Residual SAC")
    parser.add_argument("--config", default="configs/safe_residual_sac.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--online-steps", type=int)
    parser.add_argument("--lambda-anchor", type=float)
    parser.add_argument("--run-root")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(
        train_safe_residual_sac(
            args.config,
            args.seed,
            resume=args.resume,
            online_steps=args.online_steps,
            lambda_anchor=args.lambda_anchor,
            run_root=args.run_root,
            smoke=args.smoke,
        )
    )


if __name__ == "__main__":
    main()
