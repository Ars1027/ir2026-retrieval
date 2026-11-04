"""Optional stage: BM25 baseline plus a plug-in vector branch (extension point 1).

This module is only imported when ``vector.enabled`` is true in the active
method config. With the shipped ``configs/bm25.json`` the default entry never
touches it, so baseline behaviour and metrics stay byte-for-byte the same.

Responsibilities kept deliberately small:
  * run BM25 exactly as the baseline does, aggregate documents,
  * call the plug-in ``VectorRetriever.search`` and validate its output,
  * call the plug-in ``DocumentFusion.fuse`` (pass-through by default),
  * hand the resulting ranking to the shared evidence/answer assembler.
"""
from __future__ import annotations

import time

from pipeline.result_builder import assemble_result
from retrieval import interfaces
from retrieval.bm25 import BM25Retriever
from retrieval.document_rank import aggregate_documents

_CACHE = {}


def _bm25(settings):
    key = ("bm25", settings["bm25_index_dir"], settings.get("config_digest"))
    cached = _CACHE.get(key)
    if cached is None:
        started = time.monotonic()
        retriever = BM25Retriever(settings["bm25_index_dir"], settings)
        retriever.load_seconds = time.monotonic() - started
        _CACHE[key] = retriever
        cached = retriever
    return cached


def _brief_vector_hits(hits):
    keys = ("doc_id", "rank", "score")
    return [dict({k: hit[k] for k in keys}, hits=hit.get("hits", [])) for hit in hits]


def solve(question, tools, config, settings):
    query = question["question"]
    vector_config = interfaces.vector_settings(settings)
    candidate_k = int(settings.get("candidate_k") or 200)
    final_docs = int((settings.get("retrieval") or {}).get("final_docs") or 10)

    started = time.monotonic()
    dense_only = settings.get("vector", {}).get("dense_only", False)
    bm25 = None if dense_only else _bm25(settings)
    vector = interfaces.build_vector_retriever(settings)
    fusion = interfaces.build_fusion(settings)
    tools.record_retrieval(
        stage="load_indexes",
        arguments={"bm25_index_dir": settings["bm25_index_dir"],
                   "vector_index_dir": vector_config["index_dir"],
                   "vector_backend": vector_config["retriever"]},
        output={"bm25_chunks": bm25.index.n_chunks if bm25 else 0, "bm25_terms": len(bm25.index.vocab) if bm25 else 0,
                "vector": vector.describe(), "fusion": fusion.describe()},
        doc_ids=[],
        elapsed_ms=(time.monotonic() - started) * 1000,
    )

    bm25_docs = []
    if not dense_only:
        started = time.monotonic()
        hits = bm25.search(query, candidate_k)
        tools.record_retrieval(
            stage="bm25_search",
            arguments={"query": query, "top_k": candidate_k},
            output={"hits": [{k: hit[k] for k in ("chunk_id", "doc_id", "start", "end",
                                                 "score", "rank")} for hit in hits],
                    "returned": len(hits)},
            doc_ids=list(dict.fromkeys(hit["doc_id"] for hit in hits)),
            elapsed_ms=(time.monotonic() - started) * 1000,
        )

        started = time.monotonic()
        bm25_docs = aggregate_documents(hits, top_k=final_docs * 10)
        tools.record_retrieval(
            stage="document_aggregate",
            arguments={"chunks": len(hits), "top_k": final_docs * 10},
            output={"documents": [{"doc_id": doc["doc_id"], "rank": doc["rank"],
                                   "score": doc["score"]} for doc in bm25_docs]},
            doc_ids=[doc["doc_id"] for doc in bm25_docs],
            elapsed_ms=(time.monotonic() - started) * 1000,
        )

    started = time.monotonic()
    vector_docs = vector.search(query, vector_config["top_k"])
    problems = interfaces.validate_vector_hits(vector_docs, vector_config["top_k"])
    tools.record_retrieval(
        stage="vector_search",
        arguments={"query": query, "top_k": vector_config["top_k"],
                   "backend": vector_config["retriever"]},
        output={"hits": _brief_vector_hits(vector_docs) if not problems else [],
                "returned": 0 if problems else len(vector_docs),
                "errors": problems},
        doc_ids=[] if problems else [hit["doc_id"] for hit in vector_docs],
        elapsed_ms=(time.monotonic() - started) * 1000,
    )
    if problems:
        raise interfaces.RetrievalBackendError(f"invalid vector output: {problems[:3]}")

    started = time.monotonic()
    if vector_docs:
        fused = fusion.fuse(bm25_docs, vector_docs, final_docs)
        fused_problems = interfaces.validate_ranked_docs(fused)
        if fused_problems:
            raise interfaces.RetrievalBackendError(
                f"fusion returned an invalid ranking: {fused_problems[:3]}")
    else:
        fused = bm25_docs[:final_docs]
    tools.record_retrieval(
        stage="fuse",
        arguments={"fusion": (vector_config["fusion"] or "null-fusion"),
                   "top_k": final_docs},
        output={"documents": [{"doc_id": doc["doc_id"], "rank": doc["rank"],
                               "score": doc["score"], "source_ranks": doc.get("source_ranks", {})} for doc in fused],
                "used_vector_branch": bool(vector_docs)},
        doc_ids=[doc["doc_id"] for doc in fused],
        elapsed_ms=(time.monotonic() - started) * 1000,
    )
    result = assemble_result(question, fused, tools, settings, source="dense" if dense_only else "bm25+vector")
    usage = result["usage"]
    query_tokens = getattr(vector, "last_query_tokens", None)
    answer_tokens = usage.get("model_input_tokens")
    usage["model_input_tokens"] = (query_tokens + answer_tokens
                                   if query_tokens is not None and answer_tokens is not None else None)
    usage["token_notes"] += " Embedding input tokens use API usage; missing usage remains null."
    vector.close()
    return result
