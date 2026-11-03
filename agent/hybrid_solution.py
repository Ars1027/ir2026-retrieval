"""Hybrid (BM25 + dense, RRF) course adapter (member B)."""
from __future__ import annotations

import time

from pipeline.hybrid import retrieve
from pipeline.result_builder import assemble_result
from pipeline.settings import load_settings
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever

_CACHE = {}


def _retrievers(settings):
    key = (settings["bm25_index_dir"], settings["dense_index_dir"],
           settings.get("config_digest"))
    if key not in _CACHE:
        started = time.monotonic()
        bm25 = BM25Retriever(settings["bm25_index_dir"], settings)
        dense = DenseRetriever(settings["dense_index_dir"], settings)
        _CACHE[key] = (bm25, dense, time.monotonic() - started)
    return _CACHE[key]


def solve(question, tools, config):
    settings = load_settings()
    started = time.monotonic()
    bm25, dense, load_seconds = _retrievers(settings)
    tools.record_retrieval(stage="load_indexes",
                           arguments={"bm25_index_dir": settings["bm25_index_dir"],
                                      "dense_index_dir": settings["dense_index_dir"]},
                           output={"bm25_chunks": bm25.index.n_chunks,
                                   "dense_vectors": dense.index.n_vectors,
                                   "dimension": dense.index.meta.get("dimension"),
                                   "encoder": (dense.index.meta.get("encoder") or {}).get("backend")},
                           doc_ids=[],
                           elapsed_ms=(time.monotonic() - started) * 1000)
    docs = retrieve(question["question"], bm25, dense, tools, settings)
    return assemble_result(question, docs, tools, settings, source="hybrid-rrf")
