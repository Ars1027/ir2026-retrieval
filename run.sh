#!/usr/bin/env bash
# Cross-platform entry: bash run.sh --method bm25 --split dev --output output/member_b/dev_bm25
set -euo pipefail
cd "$(dirname "$0")"
exec python3 scripts/run_baseline.py "$@"
