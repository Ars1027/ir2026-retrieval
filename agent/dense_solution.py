"""Dense-only course adapter (member B, diagnostic runs).

Uses the same evidence/answer assembly as the BM25 and hybrid adapters so the
three methods differ only in retrieval.
"""
from __future__ import annotations

import time

from pipeline.result_builder import assemble_result
from pipeline.settings import load_settings
from retrieval.dense import DenseRetriever
from retrieval.document_rank import aggregate_documents

_CACHE = {}


def _retriever(settings):
    key = (settings["dense_index_dir"], settings.get("config_digest"))
    if key not in _CACHE:
        started = time.monotonic()
        retriever = DenseRetriever(settings["dense_index_dir"], settings)
        retriever.load_seconds = time.monotonic() - started
        _CACHE[key] = retriever
    return _CACHE[key]


def solve(question, tools, config):
    settings = load_settings()
    final_docs = int((settings.get("retrieval") or {}).get("final_docs") or 10)
    candidate_k = int(settings.get("candidate_k") or 200)
    query = question["question"]

    started = time.monotonic()
    retriever = _retriever(settings)
    load_ms = (time.monotonic() - started) * 1000
    tools.record_retrieval(stage="load_index",
                           arguments={"index_dir": settings["dense_index_dir"]},
                           output={"vectors": retriever.index.n_vectors,
                                   "dimension": retriever.index.meta.get("dimension"),
                                   "index_type": retriever.index.meta.get("index_type")},
                           doc_ids=[], elapsed_ms=load_ms)

    started = time.monotonic()
    hits = retriever.search(query, candidate_k)
    search_ms = (time.monotonic() - started) * 1000
    tools.record_retrieval(
        stage="dense_search",
        arguments={"query": query, "top_k": candidate_k},
        output={"hits": [{k: hit[k] for k in ("chunk_id", "doc_id", "start", "end", "score", "rank")}
                         for hit in hits], "returned": len(hits)},
        doc_ids=list(dict.fromkeys(hit["doc_id"] for hit in hits)),
        elapsed_ms=search_ms,
    )

    started = time.monotonic()
    docs = aggregate_documents(hits, top_k=final_docs)
    aggregate_ms = (time.monotonic() - started) * 1000
    tools.record_retrieval(
        stage="document_aggregate",
        arguments={"chunks": len(hits), "top_k": final_docs},
        output={"documents": [{"doc_id": d["doc_id"], "rank": d["rank"], "score": d["score"]}
                              for d in docs]},
        doc_ids=[d["doc_id"] for d in docs],
        elapsed_ms=aggregate_ms,
    )
    return assemble_result(question, docs, tools, settings, source="dense")
