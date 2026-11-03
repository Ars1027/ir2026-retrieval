"""Aggregate chunk hits into a stable document ranking.

A document enters the ranking with its best-ranked chunk; the remaining hits of
that document are kept so that real character spans survive to the evidence
stage. Document scores are the best chunk score of the same retriever, so they
stay interpretable only inside that retriever.
"""
from __future__ import annotations

from retrieval import contracts


def aggregate_documents(hits: list, top_k: int = 10) -> list:
    """Return ``list[RankedDoc]`` with ranks starting at 1 and unique documents."""
    if not isinstance(hits, list):
        raise contracts.ContractError("hits must be a list")
    if type(top_k) is not int or top_k < 1:
        raise contracts.ContractError("top_k must be a positive integer")
    grouped = {}
    order = []
    for hit in hits:
        problems = contracts.validate_hit(hit)
        if problems:
            raise contracts.ContractError(f"invalid hit: {problems}")
        doc_id = hit["doc_id"]
        if doc_id not in grouped:
            grouped[doc_id] = []
            order.append(doc_id)
        grouped[doc_id].append(hit)
    ranked = []
    for doc_id in order:
        doc_hits = sorted(grouped[doc_id], key=lambda h: (h["rank"], h["chunk_id"]))
        ranked.append((min(h["rank"] for h in doc_hits), doc_id, doc_hits))
    # Best chunk rank first; ties resolved by doc_id so repeated runs agree.
    ranked.sort(key=lambda item: (item[0], item[1]))
    out = []
    for rank, (_, doc_id, doc_hits) in enumerate(ranked[:top_k], 1):
        out.append(contracts.make_ranked_doc(doc_id, rank, doc_hits[0]["score"], doc_hits))
    return out


def best_hit(doc: dict) -> dict:
    """Highest-ranked chunk hit of a ranked document."""
    return doc["hits"][0]


def doc_ids(ranked_docs: list) -> list:
    return [doc["doc_id"] for doc in ranked_docs]
