#!/usr/bin/env bash
# Cross-platform entry: bash run.sh --method bm25 --split dev --output output/bm25/dev_bm25
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "bonus2" ]]; then
  shift
  exec bash run_bonus2.sh "$@"
fi
exec python3 scripts/run_baseline.py "$@"
