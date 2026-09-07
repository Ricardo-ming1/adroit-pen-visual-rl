from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.counterfactual.candidate_ranker import PrivilegedCandidateRanker, VisualCandidateRanker
from src.counterfactual.ranker_dataset import (
    CandidateDataset, assert_source_split_isolated, load_candidate_data, training_stats,
)
from src.counterfactual.ranker_training import (
    attach_outcome_metrics, predict_ranker, threshold_search, train_ranker,
)
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def _save_curve(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    flat = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(flat)


def _train_line(line, model, train, dev, records, config, device, run_dir, results_dir, stats):
    state, curve = train_ranker(model, train, dev, line=line, config=config["ranker"], device=device)
    model.load_state_dict(state)
    loader = DataLoader(dev, batch_size=int(config["ranker"]["batch_size"]), shuffle=False)
    predictions = predict_ranker(model, loader, line, device)
    summary = attach_outcome_metrics(threshold_search(predictions, dev, config["ranker"]), records, "dev")
    checkpoint = {
        "line": line, "model_state": state,
        "model_init": {
            "privileged_dim": train.rows[0]["privileged"].shape[-1] if line == "privileged" else None,
            "temporal_dim": train.rows[0]["temporal"].shape[-1] if line == "visual" else None,
            "candidate_count": max(r["candidate_id"] for r in train.rows) + 1,
            "hidden": int(config["ranker"]["hidden_size"]),
        },
        "stats": stats.to_dict(), "selector_threshold": summary["threshold"],
        "branch_dev_summary": summary,
        "policy_inputs": ("simulator state, phase, candidate (diagnostic only)" if line == "privileged"
                          else "frozen visual latent, proprio, base/executed action history, candidate"),
    }
    torch.save(checkpoint, run_dir / f"best_{line}_candidate_ranker.pt")
    save_json(results_dir / f"{line}_branch_dev_summary.json", summary)
    _save_curve(results_dir / f"{line}_learning_curve.csv", curve)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train privileged and frozen-visual V5 candidate rankers")
    parser.add_argument("--config", default="configs/candidate_ranker.yaml")
    args = parser.parse_args(); config = load_yaml(args.config)
    seed_everything(int(config["seed"]), bool(config["deterministic_torch"])); device = resolve_device(config["device"])
    run_dir = Path(config["run_dir"]); results_dir = Path(config["results_dir"])
    features, records, metadata = load_candidate_data(run_dir / "candidate_records.pt")
    assert_source_split_isolated(features); stats = training_stats(features)
    train = CandidateDataset(features, records, "train", stats); dev = CandidateDataset(features, records, "dev", stats)
    candidate_count = int(metadata["candidates_per_root"]); hidden = int(config["ranker"]["hidden_size"])
    privileged = PrivilegedCandidateRanker(train.rows[0]["privileged"].shape[-1], candidate_count, hidden)
    privileged_summary = _train_line("privileged", privileged, train, dev, records, config, device, run_dir, results_dir, stats)
    visual = VisualCandidateRanker(train.rows[0]["temporal"].shape[-1], candidate_count, hidden)
    visual_summary = _train_line("visual", visual, train, dev, records, config, device, run_dir, results_dir, stats)
    status = {
        "candidate_records": len(train) + len(dev), "train_records": len(train), "dev_records": len(dev),
        "privileged_pass": privileged_summary["passes_constraints"], "visual_pass": visual_summary["passes_constraints"],
        "next_stage": ("paired_closed_loop" if visual_summary["passes_constraints"] else
                       "representation_rescue" if privileged_summary["passes_constraints"] else
                       "targeted_data_expansion_or_oracle_chunk"),
    }
    save_json(results_dir / "ranker_training_status.json", status); print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
