from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.counterfactual.branch_dataset import (
    compact_root_row,
    load_root_dataset,
    save_root_dataset,
    split_roots_by_source_episode,
)
from src.counterfactual.branch_rollout import (
    collect_training_roots,
    load_frozen_actor,
    offline_action_pca,
    run_branch_pilot,
    verify_zero_branches,
)
from src.data.offline import PenOfflineData
from src.safe_residual_training import load_safe_residual_policy
from src.utils import load_yaml, package_versions, resolve_device, save_json, seed_everything


def _save_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_root_csv(path: Path, roots: list[dict[str, Any]]) -> None:
    rows = [compact_root_row(root) for root in roots]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _branch_progress(path: Path, results: list[dict[str, Any]], transitions: int) -> None:
    save_json(
        path,
        {
            "completed_roots": len(results),
            "branch_transitions": int(transitions),
            "results": results,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the counterfactual branch-supervision pilot"
    )
    parser.add_argument("--config", default="configs/counterfactual_residual.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--roots-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refresh-roots", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.smoke:
        config["run_dir"] = "runs/v4/smoke_counterfactual_pilot"
        config["roots"]["maximum_source_episodes"] = 40
        config["roots"]["checkpoint_episode_interval"] = 10
        config["roots"]["quotas"] = {key: 1 for key in config["roots"]["quotas"]}
        config["branches"]["checkpoint_root_interval"] = 1
    seed_everything(
        int(config["seed"]), bool(config.get("deterministic_torch", True))
    )
    device = resolve_device(config["device"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_yaml(run_dir / "config.yaml", config)
    data_config = load_yaml(config["data_config"])
    data = PenOfflineData(
        data_config["output_path"], frame_stack=int(data_config["frame_stack"])
    )

    vision_actor, vision_stats, vision_checkpoint = load_frozen_actor(
        config["vision_checkpoint"], device
    )
    oracle_actor, oracle_stats, oracle_checkpoint = load_frozen_actor(
        config["oracle_checkpoint"], device
    )
    safe_policy, safe_stats, safe_checkpoint = load_safe_residual_policy(
        config["safe_residual_candidate_checkpoint"], device
    )
    safe_policy.eval().requires_grad_(False)
    if vision_checkpoint["line"] != "vision" or oracle_checkpoint["line"] != "oracle":
        raise ValueError("Configured Vision/Oracle checkpoint lines are incorrect")
    if vision_stats.to_dict() != data.stats.to_dict():
        raise ValueError("Vision normalization does not match the reconstructed dataset")
    if oracle_stats.to_dict() != data.stats.to_dict():
        raise ValueError("Oracle normalization does not match the reconstructed dataset")
    if safe_stats.to_dict() != vision_stats.to_dict():
        raise ValueError("Safe Residual candidate normalization differs from Frozen AWAC")
    if int(safe_checkpoint["progress"]["online_steps"]) != 30_000:
        raise ValueError("Safe Residual proposal candidate must be the retained 30k model")

    save_json(
        run_dir / "run_meta.json",
        {
            "project": config["project"],
            "stage": "counterfactual_branch_pilot",
            "seed": int(config["seed"]),
            "environment_id": data_config["environment_id"],
            "dataset_id": data_config["dataset_id"],
            "vision_checkpoint": config["vision_checkpoint"],
            "oracle_checkpoint": config["oracle_checkpoint"],
            "safe_residual_candidate_checkpoint": config[
                "safe_residual_candidate_checkpoint"
            ],
            "simulator_state_usage": "training root restore and label generation only",
            "student_policy_inputs": "RGB, normalized hand proprio, and action history only",
            "environment_and_packages": package_versions(),
        },
    )

    root_path = run_dir / "branch_roots.pt"
    roots: list[dict[str, Any]] = []
    root_metadata: dict[str, Any] = {"source_episodes_scanned": 0}
    reference_roots: list[dict[str, Any]] = []
    if args.refresh_roots and root_path.exists():
        reference_roots, _ = load_root_dataset(root_path)
    elif args.resume and root_path.exists():
        roots, root_metadata = load_root_dataset(root_path)

    def root_checkpoint(
        partial_roots: list[dict[str, Any]], episodes_scanned: int
    ) -> None:
        partial_counts = Counter(root["root_type"] for root in partial_roots)
        save_root_dataset(
            root_path,
            partial_roots,
            {
                "source_episodes_scanned": episodes_scanned,
                "counts": dict(partial_counts),
                "complete": False,
            },
        )

    roots, root_metadata = collect_training_roots(
        vision_actor=vision_actor,
        stats=vision_stats,
        device=device,
        config=config,
        existing_roots=roots,
        start_episode_index=int(root_metadata.get("source_episodes_scanned", 0)),
        checkpoint_callback=root_checkpoint,
    )
    if reference_roots:
        reference = {root["root_id"]: root for root in reference_roots}
        if set(reference) != {root["root_id"] for root in roots}:
            raise AssertionError("Refreshed root IDs differ from branch-label roots")
        max_state_error = max(
            float(np.max(np.abs(
                root["state"]["simulator"]["qpos"]
                - reference[root["root_id"]]["state"]["simulator"]["qpos"]
            )))
            for root in roots
        )
        if max_state_error > 1e-12:
            raise AssertionError(f"Refreshed root simulator mismatch: {max_state_error}")
        root_metadata["refreshed_root_qpos_max_abs"] = max_state_error
    train_roots, dev_roots = split_roots_by_source_episode(
        roots,
        dev_fraction=float(config["roots"]["dev_fraction"]),
        split_seed=int(config["roots"]["split_seed"]),
    )
    train_ids = {root["root_id"] for root in train_roots}
    for root in roots:
        root["split"] = "train" if root["root_id"] in train_ids else "dev"
    root_metadata.update(
        train_roots=len(train_roots),
        dev_roots=len(dev_roots),
        train_source_episodes=len(
            {root["source_episode_seed"] for root in train_roots}
        ),
        dev_source_episodes=len({root["source_episode_seed"] for root in dev_roots}),
    )
    save_root_dataset(root_path, roots, root_metadata)
    save_json(run_dir / "root_collection_summary.json", root_metadata)
    _write_root_csv(run_dir / "root_index.csv", roots)

    zero_summary = verify_zero_branches(
        roots, vision_actor, vision_stats, device
    )
    save_json(run_dir / "zero_branch_validation.json", zero_summary)
    if not zero_summary["pass"]:
        raise RuntimeError(f"Zero branch did not reproduce source AWAC: {zero_summary}")
    if args.roots_only:
        print(json.dumps({"run_dir": str(run_dir), **root_metadata, **zero_summary}, indent=2))
        return

    progress_path = run_dir / "branch_results_progress.json"
    existing_results: list[dict[str, Any]] = []
    if args.resume and progress_path.exists():
        existing_results = _load_json(progress_path)["results"]
    pca = offline_action_pca(data)
    np.save(run_dir / "offline_action_pca.npy", pca)
    results, summary = run_branch_pilot(
        roots=roots,
        vision_actor=vision_actor,
        vision_stats=vision_stats,
        oracle_actor=oracle_actor,
        oracle_stats=oracle_stats,
        safe_policy=safe_policy,
        pca_directions=pca,
        device=device,
        config=config,
        existing_results=existing_results,
        progress_callback=lambda values, transitions: _branch_progress(
            progress_path, values, transitions
        ),
    )
    if int(summary["branch_transitions"]) > int(
        config["branches"]["transition_budget_max"]
    ):
        raise RuntimeError(f"Branch budget exceeded: {summary}")
    _branch_progress(progress_path, results, int(summary["branch_transitions"]))
    save_json(run_dir / "branch_pilot_summary.json", summary)
    save_json(run_dir / "branch_labels.json", {"results": results})
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
