"""Shared result assembly for the retrieval pipeline (BM25 and any future branch).

The assembler is retriever-agnostic: it receives a document ranking, reads the
source text through the course corpus, cuts verbatim quotes, asks the configured
answer strategy and returns the SPEC.md result record. Member B reuses this
module instead of copying the logic.
"""
from __future__ import annotations

import time

from ir2026 import POLICY
from pipeline import answer as answer_mod
from pipeline import evidence as evidence_mod


def _record(tools, **kwargs) -> None:
    """Log an executed stage when the caller provides course tools."""
    recorder = getattr(tools, "record_retrieval", None)
    if recorder is None:
        return
    recorder(**kwargs)


def _limits(tools) -> dict:
    limits = dict(POLICY)
    values = getattr(tools, "limits", None)
    if isinstance(values, dict):
        limits.update(values)
    return limits


def assemble_result(question, ranked_docs, tools, settings, source=None) -> dict:
    """Build the final record; never raises for missing evidence or model errors."""
    started = time.monotonic()
    limits = _limits(tools)
    settings = settings or {}
    retrieval_cfg = settings.get("retrieval") or {}
    question_id = question.get("id") if isinstance(question, dict) else None
    question_text = question.get("question", "") if isinstance(question, dict) else ""
    ranked_docs = list(ranked_docs or [])[: limits["max_doc_ids"]]
    doc_ids = [doc["doc_id"] for doc in ranked_docs]

    if not ranked_docs:
        result = {"id": question_id, "answer": "", "doc_ids": [], "evidence": [],
                  "status": "no_answer",
                  "usage": {"tool_calls": 0, "trace_events": 0, "elapsed_seconds": 0.0,
                            "model_input_tokens": 0, "model_output_tokens": 0,
                            "tool_calls_complete": True,
                            "token_notes": "No candidate documents; nothing to answer."}}
        if source:
            result["usage"]["token_notes"] += f" Retriever: {source}."
        return result

    max_quotes = min(int(retrieval_cfg.get("evidence_docs") or limits["max_evidence_quotes"]),
                     limits["max_evidence_quotes"])
    answer_cfg = settings.get("answer") or {}
    selection = {}
    if (answer_cfg.get("mode") or "extractive") == "extractive" and answer_cfg.get("span_selection"):
        selection = answer_mod.select_answer(question_text, ranked_docs, tools.corpus.text,
                                             max_docs=max_quotes)
    preferred = {selection["doc_id"]: selection["quote"]} if selection else None
    quotes = evidence_mod.build_quotes(ranked_docs, tools.corpus.text, limits,
                                       max_quotes=max_quotes,
                                       max_docs=limits["max_doc_ids"],
                                       preferred=preferred)
    if selection and selection["doc_id"] not in {q["doc_id"] for q in quotes}:
        selection = {}
    if not quotes:
        # SPEC.md rejects ranked documents without any evidence quote.
        doc_ids = []
    elapsed_ms = (time.monotonic() - started) * 1000
    _record(tools,
            stage="read_evidence",
            arguments={"doc_ids": doc_ids, "max_quotes": max_quotes},
            output=evidence_mod.quote_payload(quotes),
            doc_ids=doc_ids,
            elapsed_ms=elapsed_ms)

    answer_started = time.monotonic()
    answered = answer_mod.make_answer(question_text, quotes, settings, selection=selection)
    answer = (answered.get("answer") or "").strip()
    usage = dict(answered.get("usage") or {})
    usage.setdefault("tool_calls", 0)
    usage.setdefault("tool_calls_complete", True)
    if source:
        usage["token_notes"] = (usage.get("token_notes") or "") + f" Retriever: {source}."
    status = "ok" if answer else "no_answer"

    _record(tools,
            stage="answer",
            arguments={"mode": (settings.get("answer") or {}).get("mode", "extractive"),
                       "quotes": len(quotes)},
            output={"status": status, "answer_chars": len(answer),
                    "note": answered.get("note", "")},
            doc_ids=[],
            elapsed_ms=(time.monotonic() - answer_started) * 1000)

    result = {
        "id": question_id,
        "answer": answer,
        "doc_ids": doc_ids,
        "evidence": evidence_mod.evidence_field(quotes),
        "status": status,
        "usage": usage,
    }
    if answered.get("error"):
        result["error"] = answered["error"]
    result["usage"]["trace_events"] = len(tools.trace)
    result["usage"]["elapsed_seconds"] = round(time.monotonic() - started, 4)
    return result
