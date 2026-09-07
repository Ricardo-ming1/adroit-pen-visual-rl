from __future__ import annotations

import argparse
import json

from src.data.reconstruct import build_visual_dataset
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct Adroit Pen visual demonstrations")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_visual_dataset(load_yaml(args.config), force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
