#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python scripts/posttrain.py "$@"
else
  exec python3 scripts/posttrain.py "$@"
fi
