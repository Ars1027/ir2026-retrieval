"""Fixed hybrid retrieval pipeline: BM25 + dense -> document RRF (member B).

The pipeline is deterministic: no model chooses tools, so it runs under the
``retrieval`` method type and only emits ``retrieval`` trace events. Evidence,
answers and result assembly are delegated to member A's shared assembler.
"""
from __future__ import annotations

import time

from retrieval.document_rank import aggregate_documents
from retrieval.fusion import fuse_rrf


def _brief_hits(hits):
    keys = ("chunk_id", "doc_id", "start", "end", "score", "rank")
    return [{k: hit[k] for k in keys} for hit in hits]


def _brief_docs(docs):
    return [{"doc_id": doc["doc_id"], "rank": doc["rank"], "score": doc["score"],
             "sources": doc.get("sources")} for doc in docs]


def retrieve(query: str, bm25, dense, tools, settings: dict, record: bool = True) -> list:
    """Run both retrievers, aggregate per branch and fuse the document rankings."""
    candidate_k = int(settings.get("candidate_k") or 200)
    per_source = int(settings.get("docs_per_source") or candidate_k)
    rrf_k = int(settings.get("rrf_k") or 60)
    final_docs = int((settings.get("retrieval") or {}).get("final_docs") or 10)

    started = time.monotonic()
    bm25_hits = bm25.search(query, candidate_k) if bm25 is not None else []
    bm25_ms = (time.monotonic() - started) * 1000
    if record:
        tools.record_retrieval(stage="bm25_search",
                               arguments={"query": query, "top_k": candidate_k},
                               output={"hits": _brief_hits(bm25_hits),
                                       "returned": len(bm25_hits)},
                               doc_ids=list(dict.fromkeys(h["doc_id"] for h in bm25_hits)),
                               elapsed_ms=bm25_ms)

    started = time.monotonic()
    dense_hits = dense.search(query, candidate_k) if dense is not None else []
    dense_ms = (time.monotonic() - started) * 1000
    if record:
        tools.record_retrieval(stage="dense_search",
                               arguments={"query": query, "top_k": candidate_k},
                               output={"hits": _brief_hits(dense_hits),
                                       "returned": len(dense_hits)},
                               doc_ids=list(dict.fromkeys(h["doc_id"] for h in dense_hits)),
                               elapsed_ms=dense_ms)

    started = time.monotonic()
    bm25_docs = aggregate_documents(bm25_hits, top_k=per_source)
    dense_docs = aggregate_documents(dense_hits, top_k=per_source)
    aggregate_ms = (time.monotonic() - started) * 1000
    if record:
        tools.record_retrieval(stage="document_aggregate",
                               arguments={"bm25_chunks": len(bm25_hits),
                                          "dense_chunks": len(dense_hits),
                                          "top_k": per_source},
                               output={"bm25_documents": _brief_docs(bm25_docs),
                                       "dense_documents": _brief_docs(dense_docs)},
                               doc_ids=[d["doc_id"] for d in bm25_docs + dense_docs],
                               elapsed_ms=aggregate_ms)

    started = time.monotonic()
    fused = fuse_rrf(bm25_docs, dense_docs, k=rrf_k, top_k=final_docs)
    fuse_ms = (time.monotonic() - started) * 1000
    if record:
        tools.record_retrieval(stage="rrf_fuse",
                               arguments={"rrf_k": rrf_k, "top_k": final_docs},
                               output={"documents": _brief_docs(fused)},
                               doc_ids=[d["doc_id"] for d in fused],
                               elapsed_ms=fuse_ms)
    return fused
