"""Export only aggregate statistics; no questions, answers, IDs or traces."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

METRICS = ("gold_hit_at_1", "gold_hit_at_5", "gold_hit_at_10", "evidence_recall_at_10", "answer_em", "evaluation_scope", "full_split_queries")


def summarize(run_dir):
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    times = sorted(r["usage"]["elapsed_seconds"] for r in rows)
    result = {"queries": len(rows), "split": info["split"], "model": info.get("model"),
              "status_counts": dict(Counter(r["status"] for r in rows)),
              "mean_seconds": statistics.mean(times) if times else None,
              "p50_seconds": statistics.median(times) if times else None,
              "p95_seconds": times[math.ceil(.95 * len(times))-1] if times else None,
              "percentile_method": "p50 median; p95 nearest rank",
              "cost": None, "cost_note": "No billing evidence available",
              "build_seconds": None, "load_seconds": None,
              "index_cost_note": "Not measured by the course result schema"}
    for key in ("model_input_tokens", "model_output_tokens"):
        values = [r.get("usage", {}).get(key) for r in rows]
        result[key] = sum(values) if all(v is not None for v in values) else None
    result["token_note"] = "Totals are null if any query has unknown usage"
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    result.update({key: metrics.get(key) for key in METRICS})
    if not metrics:
        result["metric_note"] = "Official evaluation has not been run; test labels are unavailable"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

if __name__ == "__main__":
    main()
