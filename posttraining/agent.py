"""One policy loop for teacher collection, untrained student and LoRA student."""
from __future__ import annotations

import time

from ir2026 import BudgetExceeded, count_model_tool_calls, write_jsonl
from posttraining.policy import encode, messages_for, parse_action
from posttraining.tools import IndexedTools


def run_policy(question, tools, client, config, decision_path=None):
    started = time.monotonic()
    history, decisions = [], []
    tokens_in, tokens_out, invalid = 0, 0, 0
    complete_usage = True
    result = None
    max_turns = config.get("max_turns", 12)
    max_calls = min(config.get("max_tool_calls", 10), tools.limits["max_tool_calls"])
    deadline = tools.started + tools.limits["max_seconds"]
    for turn in range(max_turns):
        if time.monotonic() >= deadline:
            result = {"status": "timeout", "error": "wall-clock budget exhausted"}
            break
        messages = messages_for(question["question"], history, config.get("history_chars", 10000))
        call_started = time.monotonic()
        decision = {"turn": turn, "messages": messages, "output": None, "action_valid": False}
        try:
            text, usage = client.complete(messages, deadline)
            tokens_in += usage["input"]
            tokens_out += usage["output"]
            decision.update(output=text, usage=usage)
            action = parse_action(text)
            if action["action"] == "finish":
                result = tools.finish(action)
                decision["action_valid"] = True
                decision["elapsed_seconds"] = round(time.monotonic() - call_started, 4)
                decisions.append(decision)
                if decision_path:
                    write_jsonl(decision_path, decisions)
                break
            if count_model_tool_calls(tools.trace) >= max_calls:
                raise BudgetExceeded("configured tool-call budget")
            if action["action"] == "search":
                observation = tools.search(action["query"])
            else:
                observation = tools.read_observed(action["doc_id"], action["start"], action.get("end"))
            decision["action_valid"] = True
            observation = "TOOL_RESULT " + encode(observation)
        except BudgetExceeded as exc:
            complete_usage = False if decision["output"] is None else complete_usage
            result = {"status": "timeout" if time.monotonic() >= deadline else "budget_exceeded",
                      "error": str(exc)}
            observation = str(exc)
        except (ValueError, KeyError) as exc:
            if decision["output"] is None:
                complete_usage = False
                result = {"status": "error", "error": f"model generation failed: {exc}"}
                observation = str(exc)
            else:
                invalid += 1
                decision["error"] = str(exc)
                observation = "ACTION_ERROR " + str(exc) + ". Return a valid JSON action."
        except Exception as exc:
            complete_usage = False
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            observation = str(exc)
        decision["elapsed_seconds"] = round(time.monotonic() - call_started, 4)
        decisions.append(decision)
        if decision_path:
            write_jsonl(decision_path, decisions)
        if result:
            break
        history.append([{"role": "assistant", "content": decision["output"] or ""},
                        {"role": "user", "content": observation}])
    if result is None:
        result = {"status": "budget_exceeded", "error": "model-turn budget exhausted"}
    fallback = result["status"] in {"timeout", "error", "budget_exceeded"}
    if fallback:
        # Keep genuinely retrieved partial work, while retaining the failure status.
        partial = tools.finish({"doc_ids": tools.latest_ranking[:10], "answer": ""})
        result.update({key: partial[key] for key in ("answer", "doc_ids", "evidence")})
    result.setdefault("answer", "")
    result.setdefault("doc_ids", [])
    result.setdefault("evidence", [])
    result["id"] = question["id"]
    result["usage"] = {
        "tool_calls": count_model_tool_calls(tools.trace), "trace_events": len(tools.trace),
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "model_input_tokens": tokens_in if complete_usage else None,
        "model_output_tokens": tokens_out if complete_usage else None,
        "tool_calls_complete": True,
        "token_notes": "Local MLX: exact prompt/generation tokens per call; no API billing." if complete_usage
        else "An interrupted/failed generation has unknown token usage; completed calls are in decisions.",
        "model_calls": len(decisions), "invalid_actions": invalid,
        "fallback_ranking": fallback,
    }
    return result


def solve(question, supplied, course_config):
    import os
    from pathlib import Path
    from posttraining.client import MLXClient
    from posttraining.config import load_config
    from retrieval.bm25 import BM25Retriever

    config = load_config()
    # Runtime has no label path or data loader. Decisions are separate from tool traces.
    tools = IndexedTools(supplied, BM25Retriever(config["bm25_index_dir"]), config)
    client = MLXClient(config["model"])
    directory = os.environ.get("IR2026_DECISIONS_DIR")
    path = Path(directory) / f"{question['id']}.jsonl" if directory else None
    return run_policy(question, tools, client, config["policy"], path)
