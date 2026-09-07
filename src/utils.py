from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

# Required by deterministic CUDA matmul on CUDA >= 10.2. This must be set
# before torch creates a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic_torch
    torch.use_deterministic_algorithms(deterministic_torch, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def package_versions() -> dict[str, str]:
    import gymnasium
    import gymnasium_robotics
    import minari
    import mujoco

    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gymnasium": gymnasium.__version__,
        "gymnasium_robotics": gymnasium_robotics.__version__,
        "minari": minari.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }
