#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$paper_dir/.." && pwd)"
python3 "$repo_dir/scripts/generate_adroit_pen_report_assets.py"
cp "$repo_dir/docs/assets/adroit_pen_report/"*.png "$paper_dir/figures/"
printf 'Refreshed report figures from verified CSV/JSON summaries.\n'
