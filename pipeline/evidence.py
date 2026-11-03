"""Verbatim evidence extraction from the source documents.

Quotes are always sliced from ``Corpus.text(doc_id)``: no rewriting, no model
generation, no stitching of separate passages. Each quote is a contiguous
substring between 20 and 1200 characters.
"""
from __future__ import annotations

import re

from ir2026 import POLICY
from retrieval.document_rank import best_hit

BREAK_RE = re.compile(r"[\n.!?]")


def _limits(limits=None) -> dict:
    merged = dict(POLICY)
    if limits:
        merged.update(limits)
    return merged


def clip_span(text: str, start: int, end: int, min_chars: int, max_chars: int) -> tuple:
    """Fit a chunk span to the quote budget while staying inside the document."""
    n = len(text)
    if n == 0:
        return 0, 0
    start = min(max(0, int(start)), n)
    end = min(max(start, int(end)), n)
    if end - start < min_chars:
        end = min(n, start + max(min_chars, end - start))
        if end - start < min_chars:
            start = max(0, end - min_chars)
    if end - start > max_chars:
        end = start + max_chars
        cut = None
        for match in BREAK_RE.finditer(text, start + max(min_chars, 1), end):
            cut = match.start()
        if cut is not None and cut > start + min_chars:
            end = cut + 1
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end - start < min_chars:
        # Fall back to a clean window that still satisfies the quote budget.
        start = max(0, min(start, n - min_chars))
        end = min(n, start + min_chars)
    return start, end


def build_quotes(ranked_docs: list, text_getter, limits=None, max_quotes=None,
                 max_docs=None) -> list:
    """Return quote records ``{doc_id, start, end, text}`` for the top documents."""
    limits = _limits(limits)
    max_quotes = max_quotes or limits["max_evidence_quotes"]
    max_docs = max_docs or limits["max_doc_ids"]
    quotes = []
    seen = set()
    for doc in ranked_docs[:max_docs]:
        if len(quotes) >= max_quotes:
            break
        doc_id = doc["doc_id"]
        try:
            text = text_getter(doc_id)
        except Exception:
            continue
        if not text:
            continue
        hit = best_hit(doc)
        start, end = clip_span(text, hit["start"], hit["end"],
                               limits["min_quote_chars"], limits["max_quote_chars"])
        quote = text[start:end]
        if not limits["min_quote_chars"] <= len(quote) <= limits["max_quote_chars"]:
            continue
        if quote in seen:
            continue
        seen.add(quote)
        quotes.append({"doc_id": doc_id, "start": start, "end": end, "text": quote})
    return quotes


def quote_payload(quotes: list) -> dict:
    """Trace-friendly description of the selected quotes (no full text)."""
    return {"quotes": [{"doc_id": q["doc_id"], "start": q["start"], "end": q["end"],
                        "chars": len(q["text"])} for q in quotes]}


def evidence_field(quotes: list) -> list:
    """Result field required by SPEC.md: only doc_id and the verbatim text."""
    return [{"doc_id": q["doc_id"], "text": q["text"]} for q in quotes]
