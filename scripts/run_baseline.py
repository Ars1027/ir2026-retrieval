#!/usr/bin/env python3
"""Unified entry point for the BM25 baseline (basic task).

    python scripts/run_baseline.py --method bm25  --split dev --output output/member_b/dev_bm25 --check --evaluate
    python scripts/run_baseline.py --method final --split test --output output/member_b/test_base_final --check

``--method bm25`` runs ``agent.bm25_solution``; ``--method final`` runs the
default entry ``agent.solution``, which is the same BM25 path unless the given
config enables the reserved vector hook (``vector.enabled = true``). An external
collaborator therefore runs their backend without a new method:

    python scripts/run_baseline.py --method final --config configs/vector.example.json \
        --method-label bm25-v1+vector --split dev --output output/member_b/dev_vector

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
    # Frozen default entry: what the basic deliverable runs.
    "final": {"config": "configs/final.json", "agent": "agent.solution",
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
    settings = json.loads(config.read_text(encoding="utf-8"))
    if args.split == "test" and (not settings.get("frozen") or args.limit is not None):
        parser.error("test requires a frozen configuration and the complete split")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    env = os.environ.copy()
    env["IR2026_METHOD_CONFIG"] = str(config)
    label = args.method_label or settings.get("method", method["label"])
    cmd = [sys.executable, str(ROOT / "run.py"), "--split", args.split,
           "--agent", method["agent"], "--method-label", label,
           "--method-type", "retrieval", "--output", str(output)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    model = args.model_label if args.model_label is not None else method["model"]
    if not model and settings.get("vector", {}).get("enabled"):
        model = settings.get("embedding", {}).get("model", "text-embedding-ada-002")
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
                                     "--split", args.split, "--run-dir", str(output)] + submission +
                                    (["--questions", str(output / "questions.jsonl")] if args.limit else []), env)
    if code == 0 and args.evaluate:
        evaluate_cmd = [sys.executable, str(ROOT / "evaluate.py"), "--split", args.split,
                        "--run-dir", str(output)]
        if args.limit:
            evaluate_cmd.append("--allow-subset")
        summary["evaluate_exit"] = run(evaluate_cmd, env)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return code or summary.get("check_exit", 0) or summary.get("evaluate_exit", 0)


if __name__ == "__main__":
    raise SystemExit(main())
