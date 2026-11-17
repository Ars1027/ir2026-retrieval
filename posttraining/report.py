"""Paired full-dev reporting; exported summaries never contain course QA."""
from __future__ import annotations

import json
import statistics
from collections import Counter

from ir2026 import (Corpus, evaluate, keyed, load_policy, load_questions, read_jsonl,
                    validate_results, validate_run_info)
from posttraining.config import comparison_signature
from posttraining.data import sha256


def describe_run(root, run_dir, split="dev"):
    info = json.loads((run_dir / "run_info.json").read_text())
    policy = load_policy(root / "configs/default.json")
    questions = load_questions(root / "datas" / f"{split}.jsonl")
    results = read_jsonl(run_dir / "results.jsonl")
    errors = validate_run_info(info, policy, split, submission=True)
    errors += validate_results(questions, results, Corpus(root / "corpus"), policy["limits"],
                               run_dir / "traces", method_type=info.get("method_type"))
    if errors:
        raise ValueError(f"{run_dir.name} failed official validation: {len(errors)} errors")
    unknown_calls = sum("model_calls" not in r["usage"] for r in results)
    model_calls = None if unknown_calls else sum(r["usage"]["model_calls"] for r in results)
    unknown = sum(r["usage"].get("model_input_tokens") is None or
                  r["usage"].get("model_output_tokens") is None for r in results)
    tool_types = Counter()
    for q in questions:
        tool_types.update(e["tool"] for e in read_jsonl(run_dir / "traces" / f"{q['id']}.jsonl"))
    summary = {"queries": len(questions), "validation_errors": 0,
               "method": info["method"], "model": info["model"],
               "status_counts": dict(Counter(r["status"] for r in results)),
               "model_calls": model_calls, "mean_model_calls": model_calls / len(questions) if model_calls is not None else None,
               "queries_with_unknown_model_calls": unknown_calls,
               "tool_counts": dict(tool_types),
               "invalid_actions": sum(r["usage"].get("invalid_actions", 0) for r in results),
               "fallback_rankings": sum(r["usage"].get("fallback_ranking", False) for r in results),
               "tokens": {"input": None if unknown else sum(r["usage"]["model_input_tokens"] for r in results),
                          "output": None if unknown else sum(r["usage"]["model_output_tokens"] for r in results),
                          "queries_with_unknown_usage": unknown},
               "mean_elapsed_seconds": statistics.fmean(r["usage"]["elapsed_seconds"] for r in results),
               "results_sha256": sha256(run_dir / "results.jsonl"),
               "cost_note": "Local MLX inference; token counts include repeated history; API cost=0, electricity not measured."}
    if split != "test":
        metrics, _ = evaluate(questions, results, read_jsonl(root / "datas" / f"{split}_labels.jsonl"))
        summary["metrics"] = metrics
    else:
        summary["metrics"] = None
        summary["metrics_note"] = "Test labels are teacher-only; no test accuracy is inferred."
    return summary, results


def compare_runs(root, before, after):
    configs = [json.loads((p / "method_config.json").read_text()) for p in (before, after)]
    if comparison_signature(configs[0]) != comparison_signature(configs[1]):
        raise ValueError("before/after differ in base model, prompt budget or retrieval settings")
    if configs[0]["model"].get("adapter_path") or not configs[1]["model"].get("adapter_path"):
        raise ValueError("compare an unadapted model against its adapted version")
    infos = [json.loads((p / "run_info.json").read_text()) for p in (before, after)]
    if not infos[0].get("resources") or infos[0]["resources"] != infos[1].get("resources"):
        raise ValueError("prompt, index or base weight bytes differ across the paired runs")
    b, bresults = describe_run(root, before)
    a, aresults = describe_run(root, after)
    labels = keyed(read_jsonl(root / "datas/dev_labels.jsonl"), "dev labels")
    rb, ra = keyed(bresults, "before"), keyed(aresults, "after")
    transitions = Counter()
    for qid, label in labels.items():
        hit_before = bool(set(rb[qid]["doc_ids"][:10]) & set(label["gold_doc_ids"]))
        hit_after = bool(set(ra[qid]["doc_ids"][:10]) & set(label["gold_doc_ids"]))
        transitions[f"{int(hit_before)}_to_{int(hit_after)}"] += 1
    keys = ("gold_hit_at_1", "gold_hit_at_5", "gold_hit_at_10", "evidence_recall_at_10",
            "answer_em", "completion_rate", "mean_tool_calls", "mean_elapsed_seconds")
    return {"split": "dev", "scope": "full_split", "before": b, "after": a,
            "delta_after_minus_before": {key: a["metrics"][key] - b["metrics"][key] for key in keys},
            "paired_gold_hit_at_10": dict(transitions),
            "comparison_signature": comparison_signature(configs[0]),
            "limits": "Identical prompt, BM25 index, tools, greedy decoding and budgets; only LoRA differs.",
            "answer_quality_note": "GoldHit filtering and literal quote validity do not establish semantic answer correctness."}
