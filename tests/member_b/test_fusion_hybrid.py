"""RRF and hybrid pipeline tests (member B)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fixtures  # noqa: E402
from ir2026 import POLICY, CorpusTools  # noqa: E402
from pipeline import hybrid  # noqa: E402
from pipeline.result_builder import assemble_result  # noqa: E402
from retrieval import contracts  # noqa: E402
from retrieval.document_rank import aggregate_documents  # noqa: E402
from retrieval.fusion import fuse_rrf  # noqa: E402


def hit(chunk_id, doc_id, rank, score=1.0, start=0, end=40, source="bm25"):
    return {"chunk_id": chunk_id, "doc_id": doc_id, "start": start, "end": end,
            "score": score, "rank": rank, "source": source}


def doc(doc_id, hits):
    return contracts.make_ranked_doc(doc_id, hits[0]["rank"], hits[0]["score"], hits)


class FusionTests(unittest.TestCase):
    def test_hand_computed_rrf_scores(self):
        bm25 = [doc("a", [hit("a#0", "a", 1)]), doc("b", [hit("b#0", "b", 2)])]
        dense = [doc("b", [hit("b#0", "b", 1, source="dense")]),
                 doc("c", [hit("c#0", "c", 2, source="dense")])]
        fused = fuse_rrf(bm25, dense, k=60, top_k=3)
        self.assertEqual([d["doc_id"] for d in fused], ["b", "a", "c"])
        self.assertAlmostEqual(fused[0]["score"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused[1]["score"], 1 / 61)
        self.assertAlmostEqual(fused[2]["score"], 1 / 62)
        self.assertEqual(fused[0]["rank"], 1)
        self.assertEqual(fused[0]["sources"]["bm25"]["rank"], 2)
        self.assertEqual(fused[0]["sources"]["dense"]["rank"], 1)
        self.assertEqual(contracts.validate_ranked_docs(fused), [])

    def test_scores_are_never_added_across_retrievers(self):
        bm25 = [doc("a", [hit("a#0", "a", 1, score=100.0)])]
        dense = [doc("a", [hit("a#0", "a", 1, score=0.2, source="dense")])]
        fused = fuse_rrf(bm25, dense, k=60, top_k=5)
        self.assertAlmostEqual(fused[0]["score"], 2 / 61)
        self.assertEqual(len(fused), 1)

    def test_single_branch_and_empty_branch(self):
        bm25 = [doc("a", [hit("a#0", "a", 1)])]
        self.assertEqual([d["doc_id"] for d in fuse_rrf(bm25, [], 60, 5)], ["a"])
        self.assertEqual([d["doc_id"] for d in fuse_rrf([], bm25, 60, 5)], ["a"])
        self.assertEqual(fuse_rrf([], [], 60, 5), [])

    def test_duplicate_document_in_one_branch_contributes_once(self):
        bm25 = [doc("a", [hit("a#0", "a", 1)]), doc("a", [hit("a#1", "a", 1, start=100, end=150)])]
        fused = fuse_rrf(bm25, [], k=60, top_k=5)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0]["score"], 1 / 61)

    def test_tie_is_broken_by_doc_id(self):
        bm25 = [doc("z", [hit("z#0", "z", 1)])]
        dense = [doc("y", [hit("y#0", "y", 1, source="dense")])]
        fused = fuse_rrf(bm25, dense, k=60, top_k=5)
        self.assertEqual([d["doc_id"] for d in fused], ["y", "z"])

    def test_top_k_and_hit_provenance_are_preserved(self):
        bm25 = [doc("a", [hit("a#0", "a", 1), hit("a#1", "a", 2, start=90, end=130)])]
        dense = [doc("a", [hit("a#0", "a", 1, source="dense")]), doc("b", [hit("b#0", "b", 2, source="dense")])]
        fused = fuse_rrf(bm25, dense, k=60, top_k=1)
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(fused[0]["hits"]), 3)
        self.assertEqual({h["source"] for h in fused[0]["hits"]}, {"bm25", "dense"})
        spans = {(h["start"], h["end"]) for h in fused[0]["hits"]}
        self.assertIn((90, 130), spans)


class HybridPipelineTests(unittest.TestCase):
    def setUp(self):
        self.texts = {"a": "The Halden tidal mill was built in 1842 beside the Ostra inlet. " * 3,
                      "b": "The Ostra maritime museum opened in 1968 in a salt warehouse. " * 3}
        self.tools = CorpusTools(fixtures.FakeCorpus(self.texts), "q1", dict(POLICY),
                                 method_type="retrieval")
        self.settings = {"candidate_k": 10, "docs_per_source": 5, "rrf_k": 60,
                         "retrieval": {"final_docs": 10, "evidence_docs": 5},
                         "answer": {"mode": "extractive", "max_answer_chars": 200}}

    def test_pipeline_logs_real_stages_and_fuses(self):
        bm25 = fixtures.StubRetriever([hit("a#0", "a", 1, start=0, end=60),
                                       hit("b#0", "b", 2, start=0, end=60)])
        dense = fixtures.StubRetriever([hit("b#0", "b", 1, source="dense", start=0, end=60),
                                        hit("a#0", "a", 2, source="dense", start=0, end=60)])
        docs = hybrid.retrieve("Halden mill", bm25, dense, self.tools, self.settings)
        self.assertEqual([d["doc_id"] for d in docs], ["a", "b"])
        stages = [event["tool"] for event in self.tools.trace]
        self.assertEqual(stages, ["bm25_search", "dense_search", "document_aggregate", "rrf_fuse"])
        self.assertTrue(all(event["event_type"] == "retrieval" for event in self.tools.trace))
        self.assertEqual(bm25.calls[0][1], 10)

    def test_pipeline_survives_one_empty_branch(self):
        bm25 = fixtures.StubRetriever([])
        dense = fixtures.StubRetriever([hit("b#0", "b", 1, source="dense", start=0, end=60)])
        docs = hybrid.retrieve("salt warehouse", bm25, dense, self.tools, self.settings)
        self.assertEqual([d["doc_id"] for d in docs], ["b"])

    def test_assembler_reuse_produces_valid_result(self):
        bm25 = fixtures.StubRetriever([hit("a#0", "a", 1, start=0, end=80)])
        dense = fixtures.StubRetriever([hit("a#0", "a", 1, source="dense", start=0, end=80)])
        docs = hybrid.retrieve("When was the Halden tidal mill built?", bm25, dense,
                               self.tools, self.settings)
        result = assemble_result({"id": "q1", "question": "When was the Halden tidal mill built?"},
                                 docs, self.tools, self.settings, source="hybrid-rrf")
        self.assertIn(result["status"], {"ok", "no_answer"})
        self.assertEqual(result["doc_ids"], ["a"])
        for quote in result["evidence"]:
            self.assertIn(quote["text"], self.texts[quote["doc_id"]])
            self.assertTrue(20 <= len(quote["text"]) <= 1200)
        self.assertEqual(result["usage"]["tool_calls"], 0)

    def test_documents_are_aggregated_once_per_branch(self):
        bm25_hits = [hit("a#0", "a", 1, start=0, end=40), hit("a#1", "a", 2, start=200, end=240),
                     hit("b#0", "b", 3, start=0, end=40)]
        docs = aggregate_documents(bm25_hits, top_k=10)
        self.assertEqual([d["doc_id"] for d in docs], ["a", "b"])
        self.assertEqual(len(docs[0]["hits"]), 2)


if __name__ == "__main__":
    unittest.main()
