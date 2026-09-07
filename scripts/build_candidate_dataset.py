from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.counterfactual.branch_dataset import load_root_dataset
from src.counterfactual.branch_rollout import load_frozen_actor
from src.counterfactual.candidate_branches import (
    attach_labels,
    build_train_codebook,
    deployable_specs,
    oracle_in_bank_summary,
    run_common_horizon_branches,
)
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def _load_v4_results(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)["results"]


def _policy_root_features(root: dict[str, Any]) -> dict[str, Any]:
    simulator = root["state"]["simulator"]
    return {
        "root_id": root["root_id"],
        "source_episode_id": int(root["source_episode_seed"]),
        "root_type": root["root_type"],
        "split": root["split"],
        "rgb_history": np.asarray(root["source_observation"]["rgb"], np.uint8),
        "frozen_visual_latent_history": np.asarray(root["frozen_visual_latent_history"], np.float32),
        "proprio_history": np.asarray(root["proprio_history"], np.float32),
        "previous_base_actions": np.asarray(root["previous_base_actions"], np.float32),
        "previous_executed_actions": np.asarray(root["previous_executed_actions"], np.float32),
        "privileged": np.asarray(root["source_observation"]["privileged"], np.float32),
        "simulator_qpos": np.asarray(simulator["qpos"], np.float32),
        "simulator_qvel": np.asarray(simulator["qvel"], np.float32),
        "target_orientation": np.asarray(simulator["desired_orien"], np.float32),
        "elapsed_step": int(root["elapsed_step"]),
    }


def _save_progress(path: Path, records: list[dict[str, Any]], transitions: int) -> None:
    torch.save({"records": records, "branch_transitions": int(transitions)}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V5 common-horizon candidate records")
    parser.add_argument("--config", default="configs/candidate_ranker.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-roots", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]), bool(config["deterministic_torch"]))
    device = resolve_device(config["device"])
    run_dir = Path(config["run_dir"]); results_dir = Path(config["results_dir"])
    run_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    roots, metadata = load_root_dataset(config["v4_root_path"])
    if args.max_roots is not None:
        roots = roots[: args.max_roots]
    vision, vision_stats, vision_ckpt = load_frozen_actor(config["vision_checkpoint"], device)
    oracle, oracle_stats, oracle_ckpt = load_frozen_actor(config["oracle_checkpoint"], device)
    if vision_ckpt["line"] != "vision" or oracle_ckpt["line"] != "oracle":
        raise ValueError("Vision/Oracle checkpoint lines are incorrect")
    v4_results = _load_v4_results(config["v4_label_path"])
    codebook = build_train_codebook(v4_results, int(config["candidates"]["codebook_size"]),
                                    int(config["candidates"]["codebook_seed"]))
    pca = np.load(config["v4_pca_path"])
    specs = deployable_specs(pca, codebook)
    np.save(run_dir / "train_only_codebook.npy", codebook)
    save_json(run_dir / "candidate_bank.json", {
        "count": len(specs),
        "deployable_only": True,
        "composition": {"zero": 1, "pca_local": 8, "train_only_codebook": len(codebook)},
        "oracle_candidates_excluded": True,
        "codebook_train_source_only": True,
        "specs": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in specs],
    })

    progress_path = run_dir / "candidate_records_progress.pt"
    existing: list[dict[str, Any]] = []
    if args.resume and progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        existing = progress["records"]
    records, transitions = run_common_horizon_branches(
        roots=roots, specs=specs, vision_actor=vision, vision_stats=vision_stats,
        oracle_actor=oracle, oracle_stats=oracle_stats, device=device,
        rho=float(config["candidates"]["rho"]), gamma=float(config["candidates"]["gamma"]),
        existing=existing, checkpoint_interval=int(config["candidates"]["checkpoint_root_interval"]),
        callback=lambda values, count: _save_progress(progress_path, values, count),
    )
    attach_labels(records, config["labels"])
    root_by_id = {root["root_id"]: root for root in roots}
    dataset = {
        "root_features": [_policy_root_features(root_by_id[r["root_id"]]) for r in records],
        "records": records,
        "metadata": {
            "roots": len(records), "candidates_per_root": len(specs),
            "candidate_records": sum(len(r["candidates"]) for r in records),
            "common_horizon": True, "branch_transitions": int(transitions),
            "source_split": "V4 source-trajectory split",
            "v4_root_metadata": metadata,
        },
    }
    torch.save(dataset, run_dir / "candidate_records.pt")
    summary = oracle_in_bank_summary(records)
    summary.update(dataset["metadata"])
    save_json(results_dir / "candidate_dataset_summary.json", summary)
    _save_progress(progress_path, records, transitions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
