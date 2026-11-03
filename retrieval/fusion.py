"""Reciprocal Rank Fusion over document rankings (member B).

RRF combines ranks, never raw scores, because BM25 scores and cosine
similarities are not comparable. Each list contributes once per document:

    score(d) = sum over lists of 1 / (rrf_k + rank_list(d))

Both input rankings must already be document-level, i.e. produced by member A's
``aggregate_documents``.
"""
from __future__ import annotations

from retrieval import contracts


def fuse_rrf(bm25_docs: list, dense_docs: list, k: int = 60, top_k: int = 10) -> list:
    """Fuse two document rankings; returns ``list[RankedDoc]`` with provenance."""
    if type(k) is not int or k < 0:
        raise contracts.ContractError("rrf k must be a nonnegative integer")
    if type(top_k) is not int or top_k < 1:
        raise contracts.ContractError("top_k must be a positive integer")
    scores = {}
    provenance = {}
    hits = {}
    for source, docs in (("bm25", bm25_docs or []), ("dense", dense_docs or [])):
        seen = set()
        for position, doc in enumerate(docs, 1):
            doc_id = doc["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
            provenance.setdefault(doc_id, {})[source] = {"rank": position,
                                                         "score": float(doc["score"])}
            hits.setdefault(doc_id, []).extend(doc["hits"])
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
    out = []
    for rank, doc_id in enumerate(ordered, 1):
        merged_hits = sorted(hits[doc_id], key=lambda hit: (hit["rank"], hit["source"],
                                                            hit["chunk_id"]))
        doc = contracts.make_ranked_doc(doc_id, rank, scores[doc_id], merged_hits)
        doc["sources"] = provenance[doc_id]
        out.append(doc)
    return out
