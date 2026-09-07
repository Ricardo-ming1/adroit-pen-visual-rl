from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.build_candidate_dataset import _policy_root_features
from src.counterfactual.branch_dataset import load_root_dataset
from src.counterfactual.branch_rollout import load_frozen_actor
from src.counterfactual.candidate_branches import (
    attach_labels, deployable_specs, oracle_in_bank_summary, run_common_horizon_branches,
)
from src.counterfactual.ranker_dataset import assert_source_split_isolated
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def _paths(run_dir: Path, count: int) -> list[Path]:
    return [run_dir / f"expanded_candidate_shard_{i:02d}_of_{count:02d}.pt" for i in range(count)]


def build_shard(config, shard_index: int, shard_count: int) -> None:
    seed_everything(int(config["seed"]) + shard_index, bool(config["deterministic_torch"]))
    device = resolve_device(config["device"]); run_dir = Path(config["run_dir"])
    roots, _ = load_root_dataset(run_dir / "expanded_roots.pt")
    original_count = 240; new_roots = roots[original_count:]
    base = torch.load(run_dir / "expanded_candidate_progress.pt", map_location="cpu", weights_only=False)
    completed_ids = {r["root_id"] for r in base["records"]}
    remaining = [r for r in new_roots if r["root_id"] not in completed_ids]
    shard_roots = [r for index, r in enumerate(remaining) if index % shard_count == shard_index]
    path = _paths(run_dir, shard_count)[shard_index]; existing = []
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)["records"]
    vision, vision_stats, _ = load_frozen_actor(config["vision_checkpoint"], device)
    oracle, oracle_stats, _ = load_frozen_actor(config["oracle_checkpoint"], device)
    specs = deployable_specs(np.load(config["v4_pca_path"]), np.load(run_dir / "train_only_codebook.npy"))
    def save(values, transitions):
        torch.save({"records": values, "branch_transitions": transitions,
                    "shard_index": shard_index, "shard_count": shard_count,
                    "assigned_roots": len(shard_roots)}, path)
    records, transitions = run_common_horizon_branches(
        roots=shard_roots, specs=specs, vision_actor=vision, vision_stats=vision_stats,
        oracle_actor=oracle, oracle_stats=oracle_stats, device=device,
        rho=float(config["candidates"]["rho"]), gamma=float(config["candidates"]["gamma"]),
        existing=existing, checkpoint_interval=10, callback=save)
    save(records, transitions)
    print(json.dumps({"shard": shard_index, "assigned_roots": len(shard_roots),
                      "completed_roots": len(records), "branch_transitions": transitions}, indent=2))


def merge(config, shard_count: int) -> None:
    run_dir = Path(config["run_dir"]); results_dir = Path(config["results_dir"])
    original = torch.load(run_dir / "candidate_records.pt", map_location="cpu", weights_only=False)
    roots, root_meta = load_root_dataset(run_dir / "expanded_roots.pt"); new_roots = roots[240:]
    base = torch.load(run_dir / "expanded_candidate_progress.pt", map_location="cpu", weights_only=False)
    new_records = list(base["records"]); new_transitions = int(base["branch_transitions"])
    for path in _paths(run_dir, shard_count):
        value = torch.load(path, map_location="cpu", weights_only=False)
        new_records.extend(value["records"]); new_transitions += int(value["branch_transitions"])
    by_id = {r["root_id"]: r for r in new_records}
    expected = {r["root_id"] for r in new_roots}
    if set(by_id) != expected or len(new_records) != len(by_id):
        raise AssertionError(f"Shard merge mismatch: unique={len(by_id)} expected={len(expected)}")
    new_records = [by_id[r["root_id"]] for r in new_roots]
    attach_labels(new_records, config["labels"])
    combined_records = original["records"] + new_records
    combined_features = original["root_features"] + [_policy_root_features(r) for r in new_roots]
    assert_source_split_isolated(combined_features)
    metadata = {**original["metadata"], "roots": len(combined_records),
                "candidate_records": sum(len(r["candidates"]) for r in combined_records),
                "branch_transitions": int(original["metadata"]["branch_transitions"] + new_transitions),
                "targeted_expansion": True, "targeted_distribution": config["targeted_expansion"]["quotas"]}
    torch.save({"root_features": combined_features, "records": combined_records, "metadata": metadata},
               run_dir / "expanded_candidate_records.pt")
    summary = oracle_in_bank_summary(combined_records); summary.update(metadata); summary["root_collection"] = root_meta
    save_json(results_dir / "expanded_candidate_dataset_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel shard/merge for fixed V5 expansion branches")
    parser.add_argument("--config", default="configs/candidate_ranker.yaml")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(); config = load_yaml(args.config)
    if args.merge:
        merge(config, args.shard_count)
    elif args.shard_index is None:
        parser.error("--shard-index is required unless --merge is used")
    else:
        build_shard(config, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
