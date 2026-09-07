from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.counterfactual.candidate_ranker import PrivilegedCandidateRanker
from src.counterfactual.ranker_dataset import (
    CandidateDataset, assert_source_split_isolated, load_candidate_data, training_stats,
)
from src.counterfactual.ranker_training import (
    attach_outcome_metrics, predict_ranker, threshold_search, train_ranker,
)
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def main() -> None:
    config = load_yaml("configs/candidate_ranker.yaml")
    seed_everything(int(config["seed"]), bool(config["deterministic_torch"]))
    device = resolve_device(config["device"]); run_dir = Path(config["run_dir"])
    results_dir = Path(config["results_dir"])
    features, records, metadata = load_candidate_data(run_dir / "expanded_candidate_records.pt")
    assert_source_split_isolated(features); stats = training_stats(features)
    train = CandidateDataset(features, records, "train", stats)
    dev = CandidateDataset(features, records, "dev", stats)
    model = PrivilegedCandidateRanker(
        train.rows[0]["privileged"].shape[-1], int(metadata["candidates_per_root"]),
        int(config["ranker"]["hidden_size"]))
    state, curve = train_ranker(model, train, dev, line="privileged",
                                config=config["ranker"], device=device)
    model.load_state_dict(state)
    predictions = predict_ranker(
        model, DataLoader(dev, batch_size=int(config["ranker"]["batch_size"])),
        "privileged", device)
    summary = attach_outcome_metrics(
        threshold_search(predictions, dev, config["ranker"]), records, "dev")
    torch.save({"line": "privileged", "model_state": state, "stats": stats.to_dict(),
                "selector_threshold": summary["threshold"],
                "branch_dev_summary": summary,
                "model_init": {"privileged_dim": train.rows[0]["privileged"].shape[-1],
                               "candidate_count": int(metadata["candidates_per_root"]),
                               "hidden": int(config["ranker"]["hidden_size"])}},
               run_dir / "best_expanded_privileged_candidate_ranker.pt")
    save_json(results_dir / "expanded_privileged_branch_dev_summary.json", summary)
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in curve]
    with (results_dir / "expanded_privileged_learning_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(flat)
    status = {"roots": int(metadata["roots"]), "train_records": len(train),
              "dev_records": len(dev), "privileged_pass": summary["passes_constraints"],
              "next_stage": "train_expanded_visual" if summary["passes_constraints"] else "oracle_chunk_proposer"}
    save_json(results_dir / "expanded_privileged_status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
