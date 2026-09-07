#!/usr/bin/env python3
"""Generate the figures used by the Chinese Adroit Pen technical report.

All plotted values are loaded from committed V1--V6.1 result files.  The
figures intentionally keep evaluation banks separated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "adroit_pen_report"
COLORS = {
    "blue": "#2878B5",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "purple": "#8F63B8",
    "gray": "#7A7A7A",
    "light": "#F4F6F8",
}


def configure() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 180,
        "font.size": 10,
    })


def load_json(path: str):
    return json.loads((ROOT / path).read_text())


def load_csv(path: str):
    with (ROOT / path).open() as handle:
        return list(csv.DictReader(handle))


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def research_route() -> None:
    labels = [
        ("Exact data\nreconstruction", "数据可信"),
        ("Visual BC +\noffline AWAC", "离线起点"),
        ("RGB causal\ninterventions", "视觉确实有用"),
        ("PPO / Residual /\nbranch failures", "闭环交接困难"),
        ("Temporal state\ndistillation", "问题重写"),
        ("300-episode\nconfirmation", "独立确认"),
        ("3 seeds +\nfrozen test", "最终边界"),
    ]
    fig, ax = plt.subplots(figsize=(15.2, 3.0))
    ax.set_xlim(0, len(labels)); ax.set_ylim(0, 1); ax.axis("off")
    for i, (title, subtitle) in enumerate(labels):
        color = COLORS["blue"] if i < 3 else (COLORS["orange"] if i == 3 else COLORS["green"])
        box = FancyBboxPatch((i + 0.09, 0.26), 0.82, 0.50,
                             boxstyle="round,pad=0.025,rounding_size=0.04",
                             facecolor=color, edgecolor="none", alpha=0.94)
        ax.add_patch(box)
        ax.text(i + 0.50, 0.56, title, ha="center", va="center", color="white",
                fontsize=10.2, fontweight="bold")
        ax.text(i + 0.50, 0.15, subtitle, ha="center", va="center", color="#333333")
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((i + 0.92, 0.51), (i + 1.07, 0.51),
                                         arrowstyle="-|>", mutation_scale=13,
                                         linewidth=1.5, color="#555555"))
    ax.set_title("Adroit Pen：从少量离线示范到时序视觉状态蒸馏", fontsize=15, pad=12)
    save(fig, "01_research_route.png")


def data_ledger() -> None:
    rows = [
        ["Human demonstrations", "25 episodes", "4,975", "BC / offline AWAC"],
        ["State-distillation source", "544 episodes", "108,800", "E1 / E2 only"],
        ["D1 student-visited", "300 episodes", "60,000", "D1 only; excluded from E2"],
        ["V4 counterfactual", "240 roots", "195,403 branch", "Failure analysis"],
        ["V5 candidate branches", "1,200 roots", "2,413,490 branch", "Failure analysis"],
        ["Independent confirmation", "300 episodes", "—", "Evaluation only"],
        ["Fixed-entry hold", "200 roots", "—", "Evaluation only"],
        ["3-seed reproduction", "3 × same 100 states", "—", "Recipe evaluation"],
        ["One-shot frozen test", "200 episodes", "—", "Final evaluation"],
    ]
    fig, ax = plt.subplots(figsize=(13.8, 5.6)); ax.axis("off")
    table = ax.table(cellText=rows,
                     colLabels=["Data", "Episodes / roots", "Transitions", "Use"],
                     colWidths=[0.28, 0.23, 0.22, 0.27], cellLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DEE4")
        if row == 0:
            cell.set_facecolor(COLORS["blue"]); cell.set_text_props(color="white", weight="bold")
        elif row in (1, 2, 3):
            cell.set_facecolor("#EAF2F8")
        elif row in (4, 5):
            cell.set_facecolor("#FFF3E6")
        else:
            cell.set_facecolor("#EEF7EC")
    ax.set_title("Training and evaluation data ledger（不同用途不可混用）", fontsize=15, pad=16)
    save(fig, "02_data_ledger.png")


def reconstruction_validation() -> None:
    data = load_json("results/data_validation.json")
    checks = data["checks"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})
    counts = [data["source_states"], data["container_observations_including_replayed_final"],
              data["valid_intra_episode_transitions"]]
    labels = ["Source states", "Container obs.", "Valid transitions"]
    axes[0].bar(labels, counts, color=[COLORS["blue"], COLORS["purple"], COLORS["green"]])
    axes[0].set_ylim(0, 5600); axes[0].set_ylabel("Count")
    for i, value in enumerate(counts): axes[0].text(i, value + 80, f"{value:,}", ha="center", fontweight="bold")
    axes[0].text(1, 350, "25 episodes\n0 cross-episode transitions", ha="center",
                 bbox={"boxstyle": "round", "fc": "white", "ec": "#BBBBBB"})
    err_labels = ["Obs restore", "1-step qpos", "1-step qvel", "1-step reward", "Seq. reward"]
    errors = [checks["restored_observation_max_abs_error"], checks["step_qpos_max_abs_error"],
              checks["step_qvel_max_abs_error"], checks["step_reward_max_abs_error"],
              checks["sequential_replay_reward_max_abs_error"]]
    axes[1].barh(err_labels, errors, color=COLORS["orange"])
    axes[1].set_xscale("log"); axes[1].set_xlabel("Maximum absolute error (log scale)")
    axes[1].grid(axis="x", alpha=0.25)
    for i, value in enumerate(errors): axes[1].text(value * 1.2, i, f"{value:.2e}", va="center")
    fig.suptitle("Offline dataset reconstruction and one-step replay checks", fontsize=15)
    fig.tight_layout(); save(fig, "03_reconstruction_validation.png")


def visual_interventions() -> None:
    rows = load_csv("results/visual_interventions.csv")
    names = ["Normal", "Black", "Cross-episode", "Temporal shuffle", "Target occlusion"]
    benchmark = [100 * float(row["benchmark_pooled"]) for row in rows]
    strict = [100 * float(row["strict_pooled"]) for row in rows]
    x = np.arange(len(rows)); width = 0.35
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    b1 = ax.bar(x - width / 2, benchmark, width, label="Benchmark", color=COLORS["blue"])
    b2 = ax.bar(x + width / 2, strict, width, label="Strict", color=COLORS["orange"])
    ax.set_xticks(x, names); ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 65)
    ax.legend(frameon=False, ncol=2); ax.grid(axis="y", alpha=0.25)
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=9); ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=9)
    ax.set_title("RGB interventions on the V1 frozen Full-policy bank (1,000 episodes each)")
    fig.tight_layout(); save(fig, "04_visual_interventions.png")


def failure_mechanisms() -> None:
    rows = [
        ["Direct PPO", "Global online update", "52.7 / 19.9; no stable anchor gain", "Whole actor drift"],
        ["Residual SAC", "Small action correction", "0.2@100k: 28 / 26 vs 64 / 40", "Saturation + Q drift"],
        ["Safe Residual", "Headroom + zero anchor", "30k: 59 / 23; Q–MC bias +40.08", "Critic cannot rank tiny corrections"],
        ["Counterfactual branch", "True branch supervision", "52/240 positive; gate precision 30.8%", "Long-horizon gate/regression fails"],
        ["Candidate ranker", "Gain / harm ranking", "Privileged: 28.9% precision, 11.1% harm", "Selector remains unsafe"],
        ["Oracle takeover", "Strong 5-step teacher chunk", "205 gains / 223 harms", "Controller stitching is unstable"],
    ]
    fig, ax = plt.subplots(figsize=(15.5, 5.1)); ax.axis("off")
    table = ax.table(cellText=rows,
                     colLabels=["Route", "Intended fix", "Key evidence", "Why it stopped"],
                     colWidths=[0.16, 0.23, 0.34, 0.27], cellLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9.8); table.scale(1, 1.75)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DEE4")
        if row == 0:
            cell.set_facecolor(COLORS["red"]); cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFF7F2" if row <= 3 else "#F7F3FB")
    ax.set_title("Why action-level offline-to-online handoff was abandoned", fontsize=15, pad=16)
    save(fig, "05_failure_mechanisms.png")


def v6_ablation() -> None:
    rows = load_csv("results/v6/summary.csv")
    names = ["E0 Vision", "E-Oracle", "E1", "E2", "D1"]
    benchmark = [100 * float(row["benchmark"]) for row in rows]
    strict = [100 * float(row["strict"]) for row in rows]
    exit_rate = [100 * float(row["goal_exit_after_entry"]) for row in rows]
    x = np.arange(len(rows)); width = 0.26
    fig, ax = plt.subplots(figsize=(10.6, 5.1))
    b1 = ax.bar(x - width, benchmark, width, label="Benchmark", color=COLORS["blue"])
    b2 = ax.bar(x, strict, width, label="Strict", color=COLORS["orange"])
    b3 = ax.bar(x + width, exit_rate, width, label="Exit / entry", color=COLORS["red"], alpha=0.82)
    ax.set_xticks(x, names); ax.set_ylim(0, 90); ax.set_ylabel("Rate (%)")
    ax.legend(frameon=False, ncol=3); ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2, b3): ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8.5)
    ax.set_title("V6 ablation on the shared 100-episode development bank")
    fig.tight_layout(); save(fig, "06_v6_ablation.png")


def final_evaluation() -> None:
    confirm = load_json("results/v6_1/phase_a_confirmation300.json")["policies"]
    frozen = load_json("results/v6_1/frozen_test.json")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.7))
    labels = ["E0", "E2", "Oracle"]
    bench = [100 * confirm[k]["benchmark_success"] for k in ("e0", "e2", "oracle")]
    strict = [100 * confirm[k]["strict_stable_success"] for k in ("e0", "e2", "oracle")]
    x = np.arange(3); width = 0.35
    axes[0].bar(x - width / 2, bench, width, label="Benchmark", color=COLORS["blue"])
    axes[0].bar(x + width / 2, strict, width, label="Strict", color=COLORS["orange"])
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 85); axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Independent paired confirmation\n300 episodes, seeds 7,000,000–7,000,299")
    axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=0.25)
    labels2 = ["E0", "E2"]
    bench2 = [100 * frozen[k]["benchmark_success"] for k in ("e0", "e2")]
    strict2 = [100 * frozen[k]["strict_stable_success"] for k in ("e0", "e2")]
    x2 = np.arange(2)
    axes[1].bar(x2 - width / 2, bench2, width, label="Benchmark", color=COLORS["blue"])
    axes[1].bar(x2 + width / 2, strict2, width, label="Strict", color=COLORS["orange"])
    axes[1].set_xticks(x2, labels2); axes[1].set_ylim(0, 85)
    axes[1].set_title("One-shot paired frozen test\n200 episodes, seeds 50,000–50,199")
    axes[1].legend(frameon=False); axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        for container in ax.containers: ax.bar_label(container, fmt="%.1f", padding=2, fontsize=9)
    fig.suptitle("E2 closed-loop results (banks shown separately)", fontsize=15)
    fig.tight_layout(); save(fig, "07_confirmation_and_frozen_test.png")


def reproduction_and_hold() -> None:
    reproduction = load_json("results/v6_1/reproduction/three_seed_reproduction.json")
    fixed = load_json("results/v6_1/summary.json")["fixed_entry_hold"]["policies"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8))
    seeds = reproduction["seeds"]
    x = np.arange(3); width = 0.35
    b1 = [100 * row["benchmark_success"] for row in seeds]
    b2 = [100 * row["strict_stable_success"] for row in seeds]
    axes[0].bar(x - width / 2, b1, width, label="Benchmark", color=COLORS["blue"])
    axes[0].bar(x + width / 2, b2, width, label="Strict", color=COLORS["orange"])
    axes[0].set_xticks(x, [str(row["recipe_seed"]) for row in seeds]); axes[0].set_ylim(0, 75)
    axes[0].set_xlabel("Recipe seed (same 100 evaluation states)"); axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Three-seed recipe reproduction"); axes[0].legend(frameon=False)
    policies = ["e0", "e2", "oracle"]; labels = ["E0", "E2", "Oracle"]
    h20 = [100 * fixed[k]["twenty_step_survival"] for k in policies]
    h50 = [100 * fixed[k]["fifty_step_survival"] for k in policies]
    x2 = np.arange(3)
    axes[1].bar(x2 - width / 2, h20, width, label="20-step", color=COLORS["green"])
    axes[1].bar(x2 + width / 2, h50, width, label="50-step", color=COLORS["purple"])
    axes[1].set_xticks(x2, labels); axes[1].set_ylim(0, 75); axes[1].set_title("Fixed-entry hold, 200 common roots")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        for container in ax.containers: ax.bar_label(container, fmt="%.1f", padding=2, fontsize=9)
    fig.suptitle("Reproducibility and hold diagnostics (not pooled as one bank)", fontsize=15)
    fig.tight_layout(); save(fig, "08_reproduction_and_fixed_hold.png")


def main() -> None:
    configure()
    research_route()
    data_ledger()
    reconstruction_validation()
    visual_interventions()
    failure_mechanisms()
    v6_ablation()
    final_evaluation()
    reproduction_and_hold()
    print(f"Wrote 8 report figures to {OUT}")


if __name__ == "__main__":
    main()
