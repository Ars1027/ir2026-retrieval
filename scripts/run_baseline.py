#!/usr/bin/env python3
"""Unified entry point for the three fixed pipelines (member B).

    python scripts/run_baseline.py --method bm25 --split dev --output output/member_b/dev_bm25
    python scripts/run_baseline.py --method hybrid --split dev --output output/member_b/dev_hybrid --check --evaluate

The method configuration is exported through IR2026_METHOD_CONFIG because the
course runner has no method-config flag. Existing output directories are never
overwritten: every run needs a new or empty directory.
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

METHODS = {
    "bm25": {"config": "configs/bm25.json", "agent": "agent.bm25_solution",
             "label": "bm25-v1", "model": None},
    "dense": {"config": "configs/dense.json", "agent": "agent.dense_solution",
              "label": "dense-v1", "model": None},
    "hybrid": {"config": "configs/hybrid.json", "agent": "agent.hybrid_solution",
               "label": "hybrid-rrf-v1", "model": None},
    # Frozen default entry: what the final basic deliverable runs.
    "final": {"config": "configs/bm25.json", "agent": "agent.solution",
              "label": "bm25-v1", "model": None},
}


def run(cmd, env):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="override the method config")
    parser.add_argument("--method-label", help="override the course method label")
    parser.add_argument("--model-label", help="actual model identifier if a model is used")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    method = METHODS[args.method]
    config = Path(args.config or (ROOT / method["config"])).resolve()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    env = os.environ.copy()
    env["IR2026_METHOD_CONFIG"] = str(config)
    label = args.method_label or method["label"]
    cmd = [sys.executable, str(ROOT / "run.py"), "--split", args.split,
           "--agent", method["agent"], "--method-label", label,
           "--method-type", "retrieval", "--output", str(output)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    model = args.model_label if args.model_label is not None else method["model"]
    if model:
        cmd += ["--model-label", model]

    started = time.monotonic()
    code = run(cmd, env)
    if code == 0 and config.is_file():
        # Frozen config snapshot next to the run; not read by the course runner.
        (output / "method_config.json").write_text(config.read_text(encoding="utf-8"),
                                                   encoding="utf-8")
    summary = {"method": args.method, "label": label, "model": model,
               "config": str(config), "split": args.split, "limit": args.limit,
               "output": str(output), "runner_exit": code,
               "elapsed_seconds": round(time.monotonic() - started, 3)}
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
