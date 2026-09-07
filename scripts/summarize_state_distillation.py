from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path("results/v6")
    names = ["e0_frozen_vision_validation100", "oracle_adapter_validation100", "e1_validation100", "e2_validation100", "d1_validation100"]
    rows = []
    for name in names:
        value = json.loads((root / f"{name}.json").read_text())
        rows.append({"experiment": name, "benchmark": value["benchmark_success"],
                     "strict": value["strict_stable_success"], "goal_exit_after_entry": value["goal_exit_after_entry_rate"],
                     "hold20": value.get("consecutive_success_window_rate"), "paired_wins": value.get("paired_wins"),
                     "paired_losses": value.get("paired_losses"), "paired_ties": value.get("paired_ties")})
    with (root / "generated_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
