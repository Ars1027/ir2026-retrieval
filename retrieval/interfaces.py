"""Reserved interface for the optional vector-space branch (extension point 1).

The basic deliverable is BM25 only. This module defines the *place* where an
external collaborator plugs in a vector-space model without touching the BM25
path, the evidence/answer assembly, or the course runner adapter.

Two abstractions, both document-level so that no retriever has to know how the
other one indexes text:

``VectorRetriever``
    input : ``query: str`` (non-empty), ``top_k: int`` (>= 1)
    output: ``list[VectorHit]`` ordered by descending relevance

    ``VectorHit = {"doc_id": str, "rank": int, "score": float}``

    Rules: at most ``top_k`` items; ``rank`` runs 1, 2, 3 ... without gaps;
    ``doc_id`` is the original course corpus string and appears once; ``score``
    is a finite float where larger means more relevant; scores are only
    comparable inside one retriever.

``DocumentFusion``
    input : ``bm25_docs: list[RankedDoc]``, ``vector_docs: list[VectorHit]``,
            ``top_k: int``
    output: ``list[RankedDoc]`` (max ``top_k`` unique documents)

Nothing here runs by default: ``configs/bm25.json`` ships ``vector.enabled =
false``, so ``agent/solution.py`` executes exactly the BM25 code path.

A backend is referenced from the method config, not imported by this module:

    "vector": {
      "enabled": true,
      "retriever": "my_package.vector_backend:create_retriever",
      "fusion": "my_package.vector_backend:create_fusion",
      "top_k": 100
    }

Both specs are ``"module.path:factory"``; the factory receives the resolved
settings dict and returns an instance of the matching abstract class.
"""
from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod

from retrieval import contracts

VECTOR_HIT_FIELDS = ("doc_id", "rank", "score")
INTERFACE_VERSION = "vector-interface.v1"


class RetrievalBackendError(RuntimeError):
    """Raised when a plug-in backend is missing, malformed or returns bad data."""


def interface_descriptor() -> dict:
    """Machine-readable contract, safe to publish (no data, no paths)."""
    return {
        "version": INTERFACE_VERSION,
        "vector_retriever": {
            "method": "search(query: str, top_k: int) -> list[VectorHit]",
            "hit_fields": list(VECTOR_HIT_FIELDS),
            "constraints": ["0 < len(hits) <= top_k", "rank is 1..n without gaps",
                            "doc_id unique and unchanged", "score is finite float",
                            "larger score means more relevant"],
        },
        "document_fusion": {
            "method": "fuse(bm25_docs, vector_docs, top_k) -> list[RankedDoc]",
            "input": ["RankedDoc from retrieval.document_rank.aggregate_documents",
                      "VectorHit list from VectorRetriever.search"],
            "output": "RankedDoc list, max top_k unique doc_id, rank from 1",
        },
        "config_keys": ["vector.enabled", "vector.retriever", "vector.fusion",
                        "vector.top_k", "vector.index_dir"],
    }


class VectorRetriever(ABC):
    """Document-level vector retriever. Implementations must not touch BM25."""

    #: short identifier written into traces and records
    name = "vector"

    @abstractmethod
    def search(self, query: str, top_k: int) -> list:
        """Return up to ``top_k`` :class:`VectorHit` dicts, best first."""

    def describe(self) -> dict:
        """Non-secret metadata (backend, model, revision, index size)."""
        return {"name": self.name}

    def close(self) -> None:
        """Release resources; the course runner spawns one process per query."""


class DocumentFusion(ABC):
    """Combines the BM25 document ranking with the vector document ranking."""

    name = "fusion"

    @abstractmethod
    def fuse(self, bm25_docs: list, vector_docs: list, top_k: int) -> list:
        """Return ``list[RankedDoc]`` with at most ``top_k`` unique documents."""

    def describe(self) -> dict:
        return {"name": self.name}


class NullVectorRetriever(VectorRetriever):
    """Default backend: no vector search at all, so nothing is fused."""

    name = "null-vector"

    def search(self, query: str, top_k: int) -> list:
        return []

    def describe(self) -> dict:
        return {"name": self.name, "enabled": False,
                "note": "reserved interface only; no vector model is wired in"}


class NullFusion(DocumentFusion):
    """Default fusion: pass the BM25 ranking through untouched."""

    name = "null-fusion"

    def fuse(self, bm25_docs: list, vector_docs: list, top_k: int) -> list:
        return list(bm25_docs)[:max(int(top_k), 0)]


def validate_vector_hits(hits, top_k: int) -> list:
    """Return a list of problems; empty means the plug-in output is usable."""
    if not isinstance(hits, list):
        return ["vector hits must be a list"]
    if type(top_k) is not int or top_k < 1:
        return ["top_k must be a positive int"]
    problems = []
    if len(hits) > top_k:
        problems.append(f"returned {len(hits)} hits for top_k={top_k}")
    seen = set()
    for position, hit in enumerate(hits, 1):
        if not isinstance(hit, dict):
            problems.append(f"hit {position}: not a dict")
            continue
        missing = [field for field in VECTOR_HIT_FIELDS if field not in hit]
        if missing:
            problems.append(f"hit {position}: missing fields {missing}")
            continue
        if not isinstance(hit["doc_id"], str) or not hit["doc_id"]:
            problems.append(f"hit {position}: bad doc_id")
        elif hit["doc_id"] in seen:
            problems.append(f"hit {position}: duplicate doc_id")
        seen.add(hit["doc_id"])
        if type(hit["rank"]) is not int or hit["rank"] != position:
            problems.append(f"hit {position}: rank must equal position")
        score = hit["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            problems.append(f"hit {position}: score must be numeric")
        elif not math.isfinite(float(score)):
            problems.append(f"hit {position}: score must be finite")
    return problems


def load_spec(spec, settings: dict, kind: str):
    """Resolve ``"module.path:factory"`` and call it with ``settings``."""
    if not isinstance(spec, str) or ":" not in spec:
        raise RetrievalBackendError(
            f"{kind} spec must look like 'package.module:factory', got {spec!r}")
    module_name, _, factory_name = spec.partition(":")
    if not module_name or not factory_name:
        raise RetrievalBackendError(f"{kind} spec is incomplete: {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RetrievalBackendError(
            f"cannot import {kind} module {module_name!r}: {exc}") from exc
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise RetrievalBackendError(
            f"{module_name!r} has no callable {factory_name!r} for the {kind} backend")
    try:
        return factory(settings)
    except Exception as exc:  # plug-in failures must be explicit, never silent
        raise RetrievalBackendError(f"{kind} factory {spec!r} failed: {exc}") from exc


def vector_settings(settings: dict) -> dict:
    """Resolved ``vector`` block with defaults; ``enabled`` defaults to false."""
    raw = (settings or {}).get("vector") or {}
    if not isinstance(raw, dict):
        raise RetrievalBackendError("settings['vector'] must be an object")
    return {"enabled": bool(raw.get("enabled", False)),
            "retriever": raw.get("retriever"),
            "fusion": raw.get("fusion"),
            "top_k": int(raw.get("top_k") or settings.get("candidate_k") or 200),
            "index_dir": raw.get("index_dir") or "artifacts/vector"}


def build_vector_retriever(settings: dict) -> VectorRetriever:
    """Instantiate the configured backend, or the null backend when disabled."""
    config = vector_settings(settings)
    if not config["enabled"]:
        return NullVectorRetriever()
    if not config["retriever"]:
        raise RetrievalBackendError(
            "vector.enabled is true but vector.retriever is null; "
            "set it to 'package.module:create_retriever'")
    backend = load_spec(config["retriever"], settings, "vector retriever")
    if not isinstance(backend, VectorRetriever):
        raise RetrievalBackendError(
            f"{config['retriever']} did not return a VectorRetriever instance")
    return backend


def build_fusion(settings: dict) -> DocumentFusion:
    """Instantiate the configured fusion, or the pass-through default."""
    config = vector_settings(settings)
    if not config["enabled"]:
        return NullFusion()
    if not config["fusion"]:
        return NullFusion()
    fusion = load_spec(config["fusion"], settings, "vector fusion")
    if not isinstance(fusion, DocumentFusion):
        raise RetrievalBackendError(
            f"{config['fusion']} did not return a DocumentFusion instance")
    return fusion


def validate_ranked_docs(docs) -> list:
    """Reuse the frozen contract validation for fused output."""
    return contracts.validate_ranked_docs(docs)
