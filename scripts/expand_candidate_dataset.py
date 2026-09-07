from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.build_candidate_dataset import _policy_root_features
from src.counterfactual.branch_dataset import (
    load_root_dataset, save_root_dataset, split_roots_by_source_episode,
)
from src.counterfactual.branch_rollout import collect_training_roots, load_frozen_actor
from src.counterfactual.candidate_branches import (
    attach_labels, deployable_specs, oracle_in_bank_summary, run_common_horizon_branches,
)
from src.counterfactual.ranker_dataset import assert_source_split_isolated
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="One predefined targeted V5 root/data expansion")
    parser.add_argument("--config", default="configs/candidate_ranker.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); config = load_yaml(args.config)
    seed_everything(int(config["seed"]), bool(config["deterministic_torch"])); device = resolve_device(config["device"])
    run_dir = Path(config["run_dir"]); results_dir = Path(config["results_dir"])
    original_data = torch.load(
        run_dir / "candidate_records.pt", map_location="cpu", weights_only=False)
    original_features = original_data["root_features"]
    original_records = original_data["records"]
    original_meta = original_data["metadata"]
    if not isinstance(original_features, list) or not isinstance(original_records, list):
        raise TypeError("Unexpected candidate_records.pt schema")
    original_roots, original_root_meta = load_root_dataset(config["v4_root_path"])
    expanded_root_path = run_dir / "expanded_roots.pt"
    if args.resume and expanded_root_path.exists():
        roots, root_meta = load_root_dataset(expanded_root_path)
        start_episode = int(root_meta.get("source_episodes_scanned", config["targeted_expansion"]["source_episode_start"]))
    else:
        roots = original_roots
        start_episode = int(config["targeted_expansion"]["source_episode_start"])

    expansion = config["targeted_expansion"]
    root_config = {
        "seed": config["seed"],
        "roots": {
            "training_seed_start": expansion["training_seed_start"],
            "maximum_source_episodes": expansion["maximum_source_episodes"],
            "max_roots_per_trajectory": expansion["max_roots_per_trajectory"],
            "minimum_remaining_steps": expansion["minimum_remaining_steps"],
            "zero_verify_steps": expansion["zero_verify_steps"],
            "exit_lookahead": expansion["exit_lookahead"],
            "near_position_error": expansion["near_position_error"],
            "near_orientation_similarity": expansion["near_orientation_similarity"],
            "checkpoint_episode_interval": expansion["checkpoint_episode_interval"],
            "quotas": expansion["quotas"],
        },
        "branches": {"gamma": config["candidates"]["gamma"]},
    }
    vision, vision_stats, _ = load_frozen_actor(config["vision_checkpoint"], device)
    oracle, oracle_stats, _ = load_frozen_actor(config["oracle_checkpoint"], device)

    def save_roots(values, scanned):
        save_root_dataset(expanded_root_path, values, {
            "source_episodes_scanned": scanned, "complete": False,
            "initial_roots": len(original_roots)})

    roots, root_meta = collect_training_roots(
        vision_actor=vision, stats=vision_stats, device=device, config=root_config,
        existing_roots=roots, start_episode_index=start_episode, checkpoint_callback=save_roots)
    new_roots = roots[len(original_roots):]
    train_new, dev_new = split_roots_by_source_episode(
        new_roots, dev_fraction=float(expansion["dev_fraction"]), split_seed=int(expansion["split_seed"]))
    train_ids = {r["root_id"] for r in train_new}
    for root in new_roots:
        root["split"] = "train" if root["root_id"] in train_ids else "dev"
    root_meta.update(initial_roots=len(original_roots), new_roots=len(new_roots),
                     train_new_roots=len(train_new), dev_new_roots=len(dev_new))
    save_root_dataset(expanded_root_path, roots, root_meta)

    pca = np.load(config["v4_pca_path"]); codebook = np.load(run_dir / "train_only_codebook.npy")
    specs = deployable_specs(pca, codebook)
    progress_path = run_dir / "expanded_candidate_progress.pt"; branch_existing = []
    if args.resume and progress_path.exists():
        branch_existing = torch.load(progress_path, map_location="cpu", weights_only=False)["records"]
    def save_progress(values, transitions):
        torch.save({"records": values, "branch_transitions": transitions}, progress_path)
    new_records, branch_transitions = run_common_horizon_branches(
        roots=new_roots, specs=specs, vision_actor=vision, vision_stats=vision_stats,
        oracle_actor=oracle, oracle_stats=oracle_stats, device=device,
        rho=float(config["candidates"]["rho"]), gamma=float(config["candidates"]["gamma"]),
        existing=branch_existing, checkpoint_interval=20, callback=save_progress)
    attach_labels(new_records, config["labels"])
    combined_records = original_records + new_records
    combined_features = original_features + [_policy_root_features(r) for r in new_roots]
    assert_source_split_isolated(combined_features)
    metadata = {**original_meta, "roots": len(combined_records),
                "candidate_records": sum(len(r["candidates"]) for r in combined_records),
                "branch_transitions": int(original_meta["branch_transitions"] + branch_transitions),
                "targeted_expansion": True, "targeted_distribution": expansion["quotas"]}
    torch.save({"root_features": combined_features, "records": combined_records, "metadata": metadata},
               run_dir / "expanded_candidate_records.pt")
    summary = oracle_in_bank_summary(combined_records); summary.update(metadata); summary["root_collection"] = root_meta
    save_json(results_dir / "expanded_candidate_dataset_summary.json", summary)
    save_progress(new_records, branch_transitions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
