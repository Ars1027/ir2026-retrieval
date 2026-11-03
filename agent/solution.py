"""Default course entry point (member B).

Frozen choice for the basic deliverable: BM25 + verbatim evidence + extractive
short answers. The hybrid pipeline is available through ``configs/hybrid.json``
(``agent.hybrid_solution``) and dense diagnostics through ``configs/dense.json``.
It is not the default because the dense branch could not use the planned
semantic encoder on the machine used for these runs (no access to the model
hub), and the measured dev GoldHit@10 of that hybrid branch was lower than the
BM25 baseline. See ``records/member_b/`` for the numbers.
"""
from __future__ import annotations

from agent import bm25_solution


def solve(question, tools, config):
    return bm25_solution.solve(question, tools, config)
