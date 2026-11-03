"""Frozen data contracts shared by both members.

The structures are dictionaries so that they stay JSON-serialisable and easy to
log. This module documents the field names, provides cheap constructors and
validation helpers, and defines the on-disk artefact layout. It is owned by
member A: consumers (member B) only read these fields.

Offsets are Python character offsets into ``Corpus.text(doc_id)``. ``doc_id`` is
the original course corpus ID and is never rewritten. ``rank`` starts at 1.
``score`` is only comparable between hits produced by the same retriever.
"""
from __future__ import annotations

import hashlib
import json

CHUNK_FIELDS = ("chunk_id", "doc_id", "start", "end", "text")
HIT_FIELDS = ("chunk_id", "doc_id", "start", "end", "score", "rank", "source")
RANKED_DOC_FIELDS = ("doc_id", "rank", "score", "hits")

DEFAULT_CHUNKING = {"size": 1200, "overlap": 200, "min_chars": 120}
CHUNK_FORMAT_VERSION = "chunks.v1"
CHUNK_ID_SEPARATOR = "#"


class ContractError(ValueError):
    """Raised when a structure crosses the member boundary in the wrong shape."""


def chunk_id(doc_id: str, ordinal: int) -> str:
    """Stable chunk identifier derived only from the corpus ID and ordinal."""
    if not isinstance(doc_id, str) or not doc_id:
        raise ContractError("doc_id must be a nonempty string")
    if not isinstance(ordinal, int) or ordinal < 0:
        raise ContractError("ordinal must be a nonnegative integer")
    return f"{doc_id}{CHUNK_ID_SEPARATOR}{ordinal:05d}"


def split_chunk_id(value: str) -> tuple[str, int]:
    doc_id, _, ordinal = value.rpartition(CHUNK_ID_SEPARATOR)
    if not doc_id or not ordinal.isdigit():
        raise ContractError(f"invalid chunk_id: {value!r}")
    return doc_id, int(ordinal)


def make_chunk(doc_id: str, ordinal: int, start: int, end: int, text: str) -> dict:
    return {"chunk_id": chunk_id(doc_id, ordinal), "doc_id": doc_id,
            "start": start, "end": end, "text": text}


def make_hit(chunk: dict, score: float, rank: int, source: str) -> dict:
    return {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
            "start": chunk["start"], "end": chunk["end"],
            "score": float(score), "rank": int(rank), "source": source}


def make_ranked_doc(doc_id: str, rank: int, score: float, hits: list) -> dict:
    return {"doc_id": doc_id, "rank": int(rank), "score": float(score),
            "hits": list(hits)}


def validate_chunk(chunk: dict, text: str = None) -> list:
    """Return a list of problems; empty means valid."""
    problems = []
    if not isinstance(chunk, dict):
        return ["chunk is not a dict"]
    missing = [f for f in CHUNK_FIELDS if f not in chunk]
    if missing:
        return [f"chunk missing fields: {missing}"]
    if not isinstance(chunk["chunk_id"], str) or not chunk["chunk_id"]:
        problems.append("bad chunk_id")
    if not isinstance(chunk["doc_id"], str) or not chunk["doc_id"]:
        problems.append("bad doc_id")
    start, end, body = chunk["start"], chunk["end"], chunk["text"]
    if type(start) is not int or type(end) is not int:
        problems.append("offsets must be ints")
    elif not 0 <= start <= end:
        problems.append("invalid offset range")
    if not isinstance(body, str) or not body:
        problems.append("empty chunk text")
    if not problems and text is not None and text[start:end] != body:
        problems.append("chunk text differs from source slice")
    return problems


def validate_hit(hit: dict) -> list:
    problems = []
    if not isinstance(hit, dict):
        return ["hit is not a dict"]
    missing = [f for f in HIT_FIELDS if f not in hit]
    if missing:
        return [f"hit missing fields: {missing}"]
    if type(hit["rank"]) is not int or hit["rank"] < 1:
        problems.append("rank must be a positive int")
    if not isinstance(hit["score"], (int, float)) or isinstance(hit["score"], bool):
        problems.append("score must be numeric")
    if not isinstance(hit["source"], str) or not hit["source"]:
        problems.append("source must be a nonempty string")
    if not 0 <= hit.get("start", 0) <= hit.get("end", 0):
        problems.append("invalid span")
    return problems


def validate_ranked_docs(docs: list) -> list:
    problems = []
    if not isinstance(docs, list):
        return ["ranked_docs must be a list"]
    seen = set()
    for expected, doc in enumerate(docs, 1):
        if not isinstance(doc, dict) or set(RANKED_DOC_FIELDS) - set(doc):
            problems.append(f"doc {expected}: missing fields")
            continue
        if doc["rank"] != expected:
            problems.append(f"doc {expected}: rank must increase from 1")
        if doc["doc_id"] in seen:
            problems.append(f"doc {expected}: duplicate doc_id")
        seen.add(doc["doc_id"])
        if not doc["hits"]:
            problems.append(f"doc {expected}: empty hits")
    return problems


def fingerprint(value) -> str:
    """Deterministic short digest for configs and artefacts."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chunking_config(settings: dict) -> dict:
    """Resolve the chunking block of a method config with defaults applied."""
    raw = (settings or {}).get("chunking") or {}
    resolved = dict(DEFAULT_CHUNKING)
    for key in resolved:
        if key in raw:
            resolved[key] = int(raw[key])
    if resolved["size"] < resolved["min_chars"]:
        raise ContractError("chunking.size must be >= chunking.min_chars")
    if not 0 <= resolved["overlap"] < resolved["size"]:
        raise ContractError("chunking.overlap must be in [0, size)")
    resolved["version"] = CHUNK_FORMAT_VERSION
    return resolved
