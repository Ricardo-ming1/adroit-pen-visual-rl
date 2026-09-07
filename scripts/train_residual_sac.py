from __future__ import annotations

import argparse

from src.residual_training import train_residual_sac


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Frozen AWAC + Residual SAC")
    parser.add_argument("--config", default="configs/residual_sac.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--residual-scale", type=float)
    parser.add_argument("--online-steps", type=int)
    parser.add_argument("--run-root")
    args = parser.parse_args()
    print(
        train_residual_sac(
            args.config,
            args.seed,
            resume=args.resume,
            smoke=args.smoke,
            residual_scale=args.residual_scale,
            online_steps=args.online_steps,
            run_root=args.run_root,
        )
    )


if __name__ == "__main__":
    main()
