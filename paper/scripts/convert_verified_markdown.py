#!/usr/bin/env python3
"""Convert the verified Chinese Markdown report into modular LaTeX sections.

The script only restructures already reviewed prose. It does not read experiment
checkpoints or recompute metrics. Pandoc performs the Markdown-to-LaTeX syntax
conversion, while the explicit mapping below keeps the 17-part paper layout
stable for Git and Overleaf editing.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
SOURCE = REPO / "docs" / "adroit_pen_technical_report_zh.md"
OUT = PAPER / "sections"


def split_heading(text: str, level: int) -> tuple[str, dict[str, str]]:
    marker = "#" * level
    matches = list(re.finditer(rf"(?m)^{marker} (.+)$", text))
    prefix = text[: matches[0].start()] if matches else text
    blocks: dict[str, str] = {}
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        blocks[match.group(1).strip()] = text[match.end() : end].strip()
    return prefix, blocks


def subparts(body: str) -> tuple[str, dict[str, str]]:
    return split_heading(body, 3)


def clean(body: str) -> str:
    body = body.replace("assets/adroit_pen_report/", "figures/")
    body = re.sub(
        r"```mermaid\n.*?```",
        r"```{=latex}\n\\input{figures/temporal_distillation_architecture.tex}\n```",
        body,
        flags=re.S,
    )
    body = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", body, flags=re.S)
    body = re.sub(r"\\\((.*?)\\\)", r"$\1$", body, flags=re.S)
    body = re.sub(r"(?m)^### \d+\.\d+\s+", "## ", body)
    return re.sub(r"(?m)^### ", "## ", body).strip()


def select(parts: dict[str, str], *prefixes: str) -> str:
    selected = []
    for prefix in prefixes:
        key = next(k for k in parts if k.startswith(prefix))
        selected.append(f"## {re.sub(r'^\d+\.\d+\s*', '', key)}\n\n{parts[key]}")
    return "\n\n".join(selected)


def convert(name: str, title: str, body: str, citation: str = "") -> None:
    markdown = f"# {title}\n\n{clean(body)}\n"
    if citation:
        markdown += f"\n```{{=latex}}\n{citation}\n```\n"
    output = OUT / f"{name}.tex"
    subprocess.run(
        [
            "pandoc",
            "--from=markdown+raw_tex",
            "--to=latex",
            "--wrap=preserve",
            "--no-highlight",
            "--top-level-division=section",
            "-o",
            str(output),
        ],
        input=markdown,
        text=True,
        check=True,
    )


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    _, chapters = split_heading(text, 2)
    OUT.mkdir(parents=True, exist_ok=True)

    c4_intro, c4 = subparts(chapters[next(k for k in chapters if k.startswith("4."))])
    c5_intro, c5 = subparts(chapters[next(k for k in chapters if k.startswith("5."))])
    c11_intro, c11 = subparts(chapters[next(k for k in chapters if k.startswith("11."))])

    convert("00_abstract", "摘要", chapters["摘要"])
    convert(
        "01_background",
        "项目背景与研究问题",
        chapters[next(k for k in chapters if k.startswith("1."))],
        r"\autocite{rajeswaran2017learning,fu2020d4rl}",
    )
    convert(
        "02_task_metrics",
        "Adroit Pen 环境、观测、动作与成功指标",
        chapters[next(k for k in chapters if k.startswith("2."))],
        r"\autocite{gymnasiumrobotics2024,minari2024}",
    )
    convert("03_data_reconstruction", "人类示范的数据结构与精确状态重建", chapters[next(k for k in chapters if k.startswith("3."))])
    convert(
        "04_offline_baselines",
        "视觉 BC、AWAC 与 Oracle 基线",
        c4_intro + "\n\n" + select(c4, "4.1"),
        r"\autocite{nair2020awac}",
    )
    convert("05_visual_causality", "RGB 因果干预", select(c4, "4.2"))
    convert(
        "06_ppo_handoff",
        "Offline-to-online PPO 交接为何退化",
        select(c4, "4.3"),
        r"\autocite{schulman2017ppo}",
    )
    convert(
        "07_residual_failures",
        "Residual SAC 与 Safe Residual SAC",
        c5_intro + "\n\n" + select(c5, "5.1"),
        r"\autocite{haarnoja2018sac}",
    )
    convert("08_counterfactual", "Counterfactual branch 与 candidate ranker", select(c5, "5.2", "5.3"))
    convert("09_method_shift", "从 action refinement 转向状态重建", select(c5, "5.4"))
    convert(
        "10_temporal_distillation",
        "最终方法：Temporal Visual-State Distillation",
        chapters[next(k for k in chapters if k.startswith("6."))],
        r"\autocite{zhou2019continuity}",
    )
    convert("11_ablation", "V6 关键消融与开发结果", chapters[next(k for k in chapters if k.startswith("7."))])
    convert("12_final_evaluation", "V6.1 独立确认、复现与 frozen test", chapters[next(k for k in chapters if k.startswith("8."))])
    convert("13_lessons", "失败实验带来的技术认识", chapters[next(k for k in chapters if k.startswith("9."))])
    convert("14_limitations", "局限性与可扩展方向", chapters[next(k for k in chapters if k.startswith("10."))])
    convert("15_reproduction", "复现说明", select(c11, "11.2", "11.3", "11.4"))
    convert("16_conclusion", "结论", c11_intro + "\n\n" + select(c11, "11.1"))

    appendices = []
    for key in ("附录 A", "附录 B", "附录 C"):
        actual = next(k for k in chapters if k.startswith(key))
        appendix_title = actual.split("：", 1)[-1]
        appendices.append(f"# {appendix_title}\n\n{clean(chapters[actual])}")
    appendix_md = "\n\n".join(appendices) + "\n\n```{=latex}\n\\nocite{*}\n```\n"
    appendix_path = OUT / "90_appendices.tex"
    subprocess.run(
        ["pandoc", "--from=markdown+raw_tex", "--to=latex", "--wrap=preserve", "--no-highlight", "-o", str(appendix_path)],
        input=appendix_md,
        text=True,
        check=True,
    )

    for output in OUT.glob("*.tex"):
        latex = output.read_text(encoding="utf-8")
        latex = latex.replace(r"\includegraphics{figures/", r"\includegraphics[width=\linewidth]{figures/")
        latex = latex.replace(r"\begin{verbatim}", r"\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\small]")
        latex = latex.replace(r"\end{verbatim}", r"\end{Verbatim}")
        latex = latex.replace("≥", chr(92) + "(" + chr(92) + "geq" + chr(92) + ")").replace("≤", chr(92) + "(" + chr(92) + "leq" + chr(92) + ")")
        latex = latex.replace("的 dense-reward 配置，数据集为", "的 dense-reward 配置。" + chr(92) + "par 数据集为")
        latex = re.sub(re.escape(chr(92) + "texttt{") + "([^{}]{35,})" + re.escape("}"), lambda match: chr(92) + "nolinkurl{" + match.group(1).replace(chr(92) + "_", "_") + "}", latex)
        if output.name == "00_abstract.tex":
            latex = latex.replace(r"\section{摘要}", r"\section*{摘要}\addcontentsline{toc}{section}{摘要}", 1)
        output.write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
