#!/usr/bin/env python3
"""Aggregate a run directory into whitelisted metrics.

    python scripts/summarize_runs.py --run-dir output/member_b/dev_bm25 --split dev \
        --out records/member_b/dev_bm25_metrics.json

Only aggregate numbers are written: no question text, document IDs, answers or
quotes. Hit@k is never derived from chunk matches; it always uses
``ir2026.evaluate`` on the document ranking.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ir2026 import Corpus, evaluate, load_policy, load_questions, read_jsonl, validate_results  # noqa: E402


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _index_costs(run_dir) -> dict:
    """Build/load costs recorded in local index metadata (read-only)."""
    costs = {}
    bm25_meta = ROOT / "artifacts/member_a/bm25/meta.json"
    if bm25_meta.is_file():
        meta = json.loads(bm25_meta.read_text(encoding="utf-8"))
        costs["bm25"] = {"build_seconds": (meta.get("build_stats") or {}).get("build_seconds"),
                         "chunks": meta.get("chunks"), "terms": meta.get("terms"),
                         "params": meta.get("params"),
                         "index_bytes": sum(p.stat().st_size
                                            for p in bm25_meta.parent.glob("*") if p.is_file())}
    snapshot = run_dir / "method_config.json"
    if snapshot.is_file():
        settings = json.loads(snapshot.read_text(encoding="utf-8"))
        if settings.get("vector", {}).get("enabled"):
            vector_dir = ROOT / settings["vector"]["index_dir"]
            metadata = vector_dir / "metadata.json"
            if metadata.is_file():
                meta = json.loads(metadata.read_text(encoding="utf-8"))
                costs["vector"] = {key: meta.get(key) for key in
                    ("count", "input_tokens", "encoding_seconds", "build_wall_seconds", "index_bytes", "model_revision_note")}
    return costs


def summarize(run_dir: Path, split: str, with_labels: bool) -> dict:
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    results = read_jsonl(run_dir / "results.jsonl")
    questions = load_questions(ROOT / "datas" / f"{split}.jsonl")
    config = load_policy(ROOT / "configs/default.json")
    errors = validate_results(questions, results, Corpus(ROOT / "corpus"),
                              config["limits"], run_dir / "traces",
                              method_type=info.get("method_type"))
    latencies = [row.get("usage", {}).get("elapsed_seconds") for row in results]
    latencies = [float(value) for value in latencies if isinstance(value, (int, float))]
    statuses = {}
    for row in results:
        statuses[row.get("status")] = statuses.get(row.get("status"), 0) + 1
    tokens_in = sum((row.get("usage", {}).get("model_input_tokens") or 0) for row in results)
    tokens_out = sum((row.get("usage", {}).get("model_output_tokens") or 0) for row in results)
    unknown_tokens = sum(1 for row in results
                         if row.get("usage", {}).get("model_input_tokens") is None)
    summary = {
        "split": split,
        "method": info.get("method"),
        "method_type": info.get("method_type"),
        "model": info.get("model"),
        "protocol_version": info.get("protocol_version"),
        "queries": len(results),
        "expected_queries": len(questions),
        "validation_errors": len(errors),
        "validation_error_sample": None,  # Raw errors may contain private question IDs.
        "status_counts": statuses,
        "completion_rate": statuses.get("ok", 0) / max(len(results), 1),
        "latency_seconds": {
            "mean": round(statistics.fmean(latencies), 4) if latencies else None,
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "tokens": {"input": None if unknown_tokens else tokens_in,
                   "output": None if any(r.get("usage", {}).get("model_output_tokens") is None for r in results) else tokens_out,
                   "queries_with_unknown_tokens": unknown_tokens},
        "cost": None,
        "cost_note": "Provider billing unavailable. Tokens count successful responses only; failed/retried request billing is unknown.",
        "trace_events_total": sum(row.get("usage", {}).get("trace_events", 0) or 0 for row in results),
        "index_costs": _index_costs(run_dir),
        "citation_span_valid": errors == [],
        "notes": ["Hit@k uses document IDs from ir2026.evaluate.",
                  "Latency includes the course runner's per-question corpus initialisation."],
    }
    labels_path = ROOT / "datas" / f"{split}_labels.jsonl"
    if with_labels and labels_path.is_file():
        metrics, _ = evaluate(questions, results, read_jsonl(labels_path))
        summary["metrics"] = {key: metrics[key] for key in
                              ("queries", "gold_hit_at_1", "gold_hit_at_5", "gold_hit_at_10",
                               "evidence_recall_at_10", "answer_em", "completion_rate",
                               "mean_elapsed_seconds")}
    else:
        summary["metrics"] = None
        summary["metrics_note"] = ("No labels are available for this split; Hit@k and AnswerEM are "
                                   "not computed locally. Report status/latency only.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--out", type=Path, help="aggregate JSON path")
    parser.add_argument("--no-labels", action="store_true",
                        help="skip label-based metrics even when labels exist")
    args = parser.parse_args()
    summaries = [summarize(Path(run_dir), args.split, not args.no_labels) for run_dir in args.run_dir]
    payload = summaries[0] if len(summaries) == 1 else {"runs": summaries}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
