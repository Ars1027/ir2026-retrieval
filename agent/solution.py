"""Frozen default: BM25 + BGE-M3 document RRF; explicit BM25 config remains supported."""
from __future__ import annotations

from agent import bm25_solution
from pipeline.settings import load_settings
from retrieval import interfaces


def solve(question, tools, config):
    settings = load_settings()
    if interfaces.vector_settings(settings)["enabled"]:
        # Imported lazily so the baseline path never loads optional backends.
        from pipeline import vector_stage
        return vector_stage.solve(question, tools, config, settings)
    return bm25_solution.solve(question, tools, config)
