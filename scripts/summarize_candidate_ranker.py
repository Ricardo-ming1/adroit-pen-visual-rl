from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.utils import load_yaml, save_json


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    config = load_yaml("configs/candidate_ranker.yaml")
    run_dir = Path(config["run_dir"]); results_dir = Path(config["results_dir"])
    data = torch.load(run_dir / "expanded_candidate_records.pt", map_location="cpu", weights_only=False)
    grouped = defaultdict(list)
    for root in data["records"]:
        for candidate in root["candidates"]:
            grouped[candidate["candidate"]["name"]].append(candidate)
    rows = []
    for name, candidates in grouped.items():
        rows.append({
            "candidate": name, "kind": candidates[0]["candidate"]["kind"],
            "roots": len(candidates),
            "strong_gain": sum(x["labels"]["strong_gain"] for x in candidates),
            "harm": sum(x["labels"]["harm"] for x in candidates),
            "strong_gain_rate": sum(x["labels"]["strong_gain"] for x in candidates) / len(candidates),
            "harm_rate": sum(x["labels"]["harm"] for x in candidates) / len(candidates),
            "mean_correction_rms": float(np.mean([x["correction_rms"] for x in candidates])),
        })
    write_csv(results_dir / "candidate_outcomes.csv", sorted(rows, key=lambda x: x["candidate"]))
    augmented = json.load((results_dir / "oracle_chunk_augmented_privileged_summary.json").open())
    status = {
        "stage": "V5 candidate-conditioned conservative refinement",
        "same_bank_frozen_vision_awac": {"benchmark": 0.62, "strict": 0.34,
                                         "goal_exit_after_entry": 0.597,
                                         "seeds": [30000, 30099]},
        "same_bank_oracle_awac": {"benchmark": 0.75, "strict": 0.42},
        "common_horizon_candidate_data_complete": True,
        "initial_roots": 240, "expanded_roots": 1200,
        "fixed_candidate_records": int(data["metadata"]["candidate_records"]),
        "fixed_branch_transitions": int(data["metadata"]["branch_transitions"]),
        "data_expansion_performed": True, "representation_rescue_performed": False,
        "oracle_chunk_fallback_performed": True,
        "final_selector": {
            key: augmented[key] for key in (
                "coverage", "precision", "harm_rate", "selected_strict_rate",
                "selected_benchmark_rate", "selected_goal_exit_rate", "passes_constraints")
        },
        "paired_closed_loop_validation_run": False,
        "dagger_rounds_run": 0, "confirmation_run": False,
        "three_seed_run": False, "frozen_test_run": False,
        "selected_deployable_checkpoint": config["vision_checkpoint"],
        "v5_checkpoint_status": "diagnostic only; no selector met deployment constraints",
        "stop_reason": "Even the Oracle-chunk-augmented privileged selector had 28.9% precision and 11.1% harm.",
    }
    save_json(results_dir / "experiment_status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
