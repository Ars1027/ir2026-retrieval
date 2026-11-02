#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 run_demo_report.py --limit 3 "$@"
