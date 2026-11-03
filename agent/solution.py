"""Default course entry point: BM25 baseline, with a reserved vector seam.

``solve(question, tools, config)`` executes the frozen BM25 baseline. The only
extension point is ``pipeline.vector_stage``, which runs when the active method
config sets ``vector.enabled: true``; ``configs/bm25.json`` ships it as ``false``,
so the executed code path, trace stages and metrics are identical to
``agent.bm25_solution``.

Interface contract for the vector-space extension is documented in
``retrieval/interfaces.py`` and ``configs/vector.example.json``.
"""
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
