"""Resume independent course queries without retrying completed failures.

The unchanged course worker/stop_process enforce the original per-query budget.
Only an externally interrupted query without a result is rerun; its partial
observations/decisions are archived first, never overwritten or graded as a
completed query.
"""
from __future__ import annotations

import fcntl
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

from ir2026 import (Corpus, DataError, count_model_tool_calls, load_policy,
                    load_questions, read_jsonl, validate_results,
                    validate_run_info, write_jsonl)
from posttraining.data import sha256
from posttraining.config import comparison_signature
from run import stop_process, worker


def remaining_questions(questions, results):
    ids = [r["id"] for r in results]
    if ids != [q["id"] for q in questions[:len(results)]]:
        raise ValueError("completed records are not the exact official question prefix")
    return questions[len(results):]


def resume_run(root, output, settings, resources, label, resource_reference=None):
    root, output = Path(root), Path(output)
    info = json.loads((output / "run_info.json").read_text())
    policy = load_policy(root / "configs/default.json")
    split = info["split"]
    errors = validate_run_info(info, policy, split, submission=True)
    if errors:
        raise ValueError("invalid run metadata: " + "; ".join(errors))
    if info.get("model") != label or info.get("method") != settings["method"]:
        raise ValueError("model/adapter or method differs from the interrupted run")
    if info.get("agent_module") != "agent.posttrained_solution":
        raise ValueError("resume only supports the posttraining course adapter")
    expected = info.get("resources")
    if expected is None:
        if resource_reference is None:
            raise ValueError("old run needs --resource-reference for prompt/index/base checks")
        expected = json.loads(Path(resource_reference).read_text())["resources"]
        reference_settings = json.loads((Path(resource_reference).parent / "method_config.json").read_text())
        if comparison_signature(reference_settings) != comparison_signature(settings):
            raise ValueError("experimental controls differ from the recorded reference")
    if expected != resources:
        raise ValueError("prompt, index or base weights changed since the recorded reference")
    snapshot = output / "method_config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != settings:
        raise ValueError("method settings changed since the interrupted run")
    questions = load_questions(root / "datas" / f"{split}.jsonl")
    if read_jsonl(output / "questions.jsonl") != questions:
        raise ValueError("question snapshot is not the complete unchanged official split")
    # flock is released by the OS even when the user interrupts a second time.
    with (output / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = read_jsonl(output / "results.jsonl")
        pending = remaining_questions(questions, results)
        corpus = Corpus(root / "corpus")
        errors = validate_results(questions[:len(results)], results, corpus, policy["limits"],
                                  output / "traces", method_type="agent")
        if errors:
            raise ValueError(f"completed records failed validation: {len(errors)} errors")
        initial_results = list(results)
        history_path = output / "resume_history.jsonl"
        history = read_jsonl(history_path) if history_path.exists() else []
        attempt = len(history) + 1
        saved = []
        for question in pending:
            qid = question["id"]
            for folder in ("traces", "decisions"):
                path = output / folder / f"{qid}.jsonl"
                if path.exists():
                    target = output / "interrupted_attempts" / f"resume_{attempt:03d}" / folder / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(path, target)
                    saved.append(str(target.relative_to(output)))
        record = {"attempt": attempt, "completed_before": len(results), "pending": len(pending),
                  "results_sha256_before": sha256(output / "results.jsonl"),
                  "archived_partial_files": saved, "status": "running",
                  "note": "Completed failures retained; user-interrupted unfinished attempts archived separately."}
        history.append(record)
        write_jsonl(history_path, history)
        snapshot.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
        config = {**policy, "method_type": "agent", "model": label}
        ctx = mp.get_context("spawn")
        started = time.monotonic()
        for question in pending:
            receiving, sending = ctx.Pipe(duplex=False)
            trace_path = output / "traces" / f"{question['id']}.jsonl"
            process = ctx.Process(target=worker, args=(sending, info["agent_module"], question,
                                                      root / "corpus", config, trace_path))
            process.start()
            sending.close()
            try:
                result, trace = None, []
                if receiving.poll(policy["limits"]["max_seconds"]):
                    try:
                        result, trace = receiving.recv()
                    except EOFError:
                        pass
                if result is None:
                    timed_out = process.is_alive()
                    stop_process(process)
                    if trace_path.exists():
                        try:
                            trace = read_jsonl(trace_path)
                        except (DataError, OSError):
                            trace = []
                    result = {"id": question["id"], "answer": "", "doc_ids": [], "evidence": [],
                              "status": "timeout" if timed_out else "error",
                              "error": "Worker timed out or exited before returning a record.",
                              "usage": {"tool_calls": count_model_tool_calls(trace), "trace_events": len(trace),
                                        "tool_calls_complete": False,
                                        "elapsed_seconds": policy["limits"]["max_seconds"] if timed_out else 0,
                                        "model_input_tokens": None, "model_output_tokens": None,
                                        "token_notes": "Worker did not return complete usage; persisted events only."}}
            finally:
                receiving.close()
                process.join(1)
                stop_process(process)
            results.append(result)
            write_jsonl(trace_path, trace)
            write_jsonl(output / "results.jsonl", results)
            print(f"{len(results)}/{len(questions)} id={question['id']} status={result['status']}", flush=True)
        if results[:len(initial_results)] != initial_results:
            raise RuntimeError("resume changed completed records")
        record.update(status="completed", completed_after=len(results),
                      resumed_wall_seconds=round(time.monotonic() - started, 3),
                      retained_records_unchanged=True)
        write_jsonl(history_path, history)
