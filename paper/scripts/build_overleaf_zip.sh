#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="$paper_dir/dist/adroit_pen_overleaf.zip"
mkdir -p "$paper_dir/dist"
rm -f "$archive"
cd "$paper_dir"
zip -q -r "$archive" main.tex sections figures tables references.bib .latexmkrc \
  -x '*.DS_Store' '__MACOSX/*' '*.pyc' '__pycache__/*'
unzip -tq "$archive"
printf 'Created %s\n' "$archive"
