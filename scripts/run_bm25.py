#!/usr/bin/env python3
"""Run the BM25 method through the unmodified course runner (member A).

The method configuration is passed with the IR2026_METHOD_CONFIG environment
variable because the course runner has no method-config CLI flag.

Examples:
    python scripts/run_bm25.py --split train --limit 5 --output output/member_a/train_smoke
    python scripts/run_bm25.py --split dev --output output/member_a/dev_bm25 --check --evaluate
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, env):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bm25.json")
    parser.add_argument("--agent", default="agent.bm25_solution")
    parser.add_argument("--method-label", default="bm25-v1")
    parser.add_argument("--check", action="store_true", help="run check_results.py afterwards")
    parser.add_argument("--evaluate", action="store_true", help="run evaluate.py afterwards")
    args = parser.parse_args()

    env = os.environ.copy()
    env["IR2026_METHOD_CONFIG"] = str(Path(args.config).resolve())
    output = Path(args.output)
    cmd = [sys.executable, str(ROOT / "run.py"), "--split", args.split,
           "--agent", args.agent, "--method-label", args.method_label,
           "--method-type", "retrieval", "--output", str(output)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    started = time.monotonic()
    code = run(cmd, env)
    summary = {"runner_exit": code, "output": str(output),
               "elapsed_seconds": round(time.monotonic() - started, 3),
               "config": str(Path(args.config).resolve())}
    if code == 0 and args.check:
        submission = ["--submission"] if args.split in ("dev", "test") and not args.limit else []
        summary["check_exit"] = run([sys.executable, str(ROOT / "check_results.py"),
                                     "--split", args.split, "--run-dir", str(output)] + submission, env)
    if code == 0 and args.evaluate:
        evaluate_cmd = [sys.executable, str(ROOT / "evaluate.py"), "--split", args.split,
                        "--run-dir", str(output)]
        if args.limit:
            evaluate_cmd.append("--allow-subset")
        summary["evaluate_exit"] = run(evaluate_cmd, env)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
