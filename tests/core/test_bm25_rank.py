"""BM25 and document aggregation tests (core)."""
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

from retrieval import bm25, contracts  # noqa: E402
from retrieval.document_rank import aggregate_documents  # noqa: E402
import fixtures  # noqa: E402


class BM25Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = fixtures.write_corpus(Path(cls.tmp.name) / "corpus")
        cls.chunks_dir, cls.index_dir, cls.stats = fixtures.build_test_index(
            root, Path(cls.tmp.name) / "index", k1=0.9, b=0.4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def retriever(self):
        return bm25.BM25Index(self.index_dir).load()

    def test_entity_query_finds_the_right_document(self):
        hits = self.retriever().search_hits("Zarnia Observatory founder", 10)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["doc_id"], "ZA-1")
        self.assertEqual([h["rank"] for h in hits], list(range(1, len(hits) + 1)))
        for hit in hits:
            self.assertEqual(contracts.validate_hit(hit), [])

    def test_year_query_is_preserved(self):
        hits = self.retriever().search_hits("1788 customs ledger", 10)
        self.assertTrue(hits)
        self.assertIn("ZA-2", [h["doc_id"] for h in hits])
        self.assertIn("ZA-4", [h["doc_id"] for h in hits])

    def test_empty_and_unmatched_queries_are_deterministic(self):
        index = self.retriever()
        self.assertEqual(index.search_hits("", 10), [])
        self.assertEqual(index.search_hits("   ", 10), [])
        self.assertEqual(index.search_hits("zzzz-unlisted-term", 10), [])

    def test_reload_gives_identical_ranking(self):
        first = self.retriever().search_hits("Kestrel Ridge plateau survey", 10)
        second = bm25.BM25Index(self.index_dir).load().search_hits("Kestrel Ridge plateau survey", 10)
        self.assertEqual([(h["chunk_id"], h["score"]) for h in first],
                         [(h["chunk_id"], h["score"]) for h in second])

    def test_top_k_is_respected(self):
        hits = self.retriever().search_hits("archive ledger observatory", 2)
        self.assertLessEqual(len(hits), 2)

    def test_index_metadata_records_params_and_source(self):
        meta = bm25.BM25Index(self.index_dir).load().meta
        self.assertEqual(meta["format"], bm25.INDEX_VERSION)
        self.assertIn("chunking_fingerprint", meta)
        self.assertIn("build_stats", meta)
        self.assertIn("corpus", meta)


class AggregationTests(unittest.TestCase):
    def hit(self, chunk_id, doc_id, rank, score, start=0, end=50):
        return {"chunk_id": chunk_id, "doc_id": doc_id, "start": start, "end": end,
                "score": score, "rank": rank, "source": "bm25"}

    def test_multiple_chunks_of_one_document_occupy_one_slot(self):
        hits = [self.hit("d1#00000", "d1", 1, 9.0),
                self.hit("d1#00001", "d1", 2, 8.0),
                self.hit("d2#00000", "d2", 3, 7.0),
                self.hit("d3#00000", "d3", 4, 6.0)]
        docs = aggregate_documents(hits, top_k=2)
        self.assertEqual([d["doc_id"] for d in docs], ["d1", "d2"])
        self.assertEqual([d["rank"] for d in docs], [1, 2])
        self.assertEqual(len(docs[0]["hits"]), 2)
        self.assertEqual(docs[0]["score"], 9.0)
        self.assertEqual(contracts.validate_ranked_docs(docs), [])

    def test_equal_best_rank_is_resolved_by_doc_id(self):
        hits = [self.hit("b#00000", "b", 1, 5.0), self.hit("a#00000", "a", 1, 5.0)]
        docs = aggregate_documents(hits, top_k=10)
        self.assertEqual([d["doc_id"] for d in docs], ["a", "b"])

    def test_spans_survive_aggregation(self):
        hits = [self.hit("d1#00000", "d1", 1, 3.0, 100, 300)]
        docs = aggregate_documents(hits, top_k=10)
        self.assertEqual((docs[0]["hits"][0]["start"], docs[0]["hits"][0]["end"]), (100, 300))

    def test_invalid_hits_are_rejected(self):
        with self.assertRaises(contracts.ContractError):
            aggregate_documents([{"doc_id": "d1"}], top_k=5)


if __name__ == "__main__":
    unittest.main()
