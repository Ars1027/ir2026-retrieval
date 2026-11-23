"""BM25 course adapter (core).

``solve(question, tools, config)`` is the only entry point the course runner
needs. It loads the BM25 index once per process (the runner spawns one process
per question on Windows), records the stages that actually ran, and delegates
evidence/answer assembly to ``pipeline.result_builder``.
"""
from __future__ import annotations

import time

from pipeline.result_builder import assemble_result
from pipeline.settings import load_settings
from retrieval.bm25 import BM25Retriever
from retrieval.document_rank import aggregate_documents

_CACHE = {}


def _retriever(settings):
    key = (settings["bm25_index_dir"], settings.get("config_digest"))
    cached = _CACHE.get(key)
    if cached is None:
        started = time.monotonic()
        retriever = BM25Retriever(settings["bm25_index_dir"], settings)
        retriever.load_seconds = time.monotonic() - started
        _CACHE[key] = retriever
        cached = retriever
    return cached


def _serialise_hits(hits):
    keys = ("chunk_id", "doc_id", "start", "end", "score", "rank")
    return [{k: hit[k] for k in keys} for hit in hits]


def solve(question, tools, config):
    settings = load_settings()
    final_docs = settings["retrieval"]["final_docs"]
    candidate_k = settings["candidate_k"]
    query = question["question"]

    load_started = time.monotonic()
    retriever = _retriever(settings)
    load_ms = (time.monotonic() - load_started) * 1000
    tools.record_retrieval(stage="load_index", arguments={"index_dir": settings["bm25_index_dir"]},
                           output={"chunks": retriever.index.n_chunks,
                                   "terms": len(retriever.index.vocab)},
                           doc_ids=[], elapsed_ms=load_ms)

    started = time.monotonic()
    hits = retriever.search(query, candidate_k)
    search_ms = (time.monotonic() - started) * 1000
    tools.record_retrieval(
        stage="bm25_search",
        arguments={"query": query, "top_k": candidate_k},
        output={"hits": _serialise_hits(hits), "returned": len(hits)},
        doc_ids=list(dict.fromkeys(hit["doc_id"] for hit in hits)),
        elapsed_ms=search_ms,
    )

    passage = settings["passage_rerank"]
    rerank_enabled = passage["mode"] != "off"
    depth = passage["candidate_docs"] if rerank_enabled else final_docs
    started = time.monotonic()
    docs = aggregate_documents(hits, top_k=depth)
    aggregate_ms = (time.monotonic() - started) * 1000
    tools.record_retrieval(
        stage="document_aggregate",
        arguments={"chunks": len(hits), "top_k": depth},
        output={"documents": [{"doc_id": d["doc_id"], "rank": d["rank"],
                               "score": d["score"], "best_hit": d["hits"][0]["chunk_id"]}
                              for d in docs]},
        doc_ids=[d["doc_id"] for d in docs],
        elapsed_ms=aggregate_ms,
    )

    if rerank_enabled:
        from retrieval.passage_rank import rerank_documents

        started = time.monotonic()
        options = dict(passage, use_stopwords=settings["bm25"]["use_stopwords"])
        ranked, details = rerank_documents(query, docs, tools.corpus.text, options)
        docs = ranked[:final_docs]
        tools.record_retrieval(
            stage="passage_rerank",
            arguments=passage,
            output={"documents": details, "returned": len(docs)},
            doc_ids=[doc["doc_id"] for doc in ranked],
            elapsed_ms=(time.monotonic() - started) * 1000,
        )
    return assemble_result(question, docs, tools, settings, source="bm25")
