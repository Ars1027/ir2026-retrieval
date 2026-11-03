"""Encoder and dense-index tests (member B).

Run from the project root:
    python -m unittest discover -s tests/member_b -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

import fixtures  # noqa: E402
from retrieval import dense  # noqa: E402
from retrieval.document_rank import aggregate_documents  # noqa: E402
from retrieval.embedding import Encoder  # noqa: E402


class EncoderTests(unittest.TestCase):
    def test_hashing_encoder_is_deterministic_and_normalised(self):
        config = fixtures.hashing_encoder(48)
        first = Encoder(config).encode_documents(["Halden tidal mill", "Ostra inlet"])
        second = Encoder(config).encode_documents(["Halden tidal mill", "Ostra inlet"])
        self.assertEqual(first.shape, (2, 48))
        np.testing.assert_allclose(first, second)
        norms = np.linalg.norm(first, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), rtol=1e-5)

    def test_similar_texts_score_higher_than_unrelated_texts(self):
        encoder = Encoder(fixtures.hashing_encoder(64))
        vectors = encoder.encode_documents([
            "Halden tidal mill built in 1842 beside the Ostra inlet",
            "Halden mill beside the Ostra inlet",
            "completely unrelated paragraph about shipping insurance",
        ])
        related = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        self.assertGreater(related, unrelated)

    def test_query_instruction_is_applied_for_hashing_backend(self):
        encoder = Encoder(fixtures.hashing_encoder(16))
        query = encoder.encode_query("Halden mill")
        self.assertEqual(query.shape, (16,))
        self.assertAlmostEqual(float(np.linalg.norm(query)), 1.0, places=5)

    def test_dimension_guard(self):
        with self.assertRaises(Exception):
            Encoder({"backend": "hashing", "hashing": {"dimension": 4}})


class DenseIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = fixtures.write_corpus(Path(cls.tmp.name) / "corpus")
        cls.chunks_dir = Path(cls.tmp.name) / "chunks"
        fixtures.write_chunk_store(root, cls.chunks_dir)
        cls.index_dir = Path(cls.tmp.name) / "dense"
        cls.meta = dense.build_index(cls.chunks_dir, cls.index_dir,
                                     settings={"dense": {"encoder": fixtures.hashing_encoder(32)}},
                                     encoder=Encoder(fixtures.hashing_encoder(32)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_meta_records_encoder_and_chunking_fingerprint(self):
        meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["format"], dense.INDEX_VERSION)
        self.assertEqual(meta["dimension"], 32)
        self.assertEqual(meta["encoder"]["backend"], "hashing")
        self.assertTrue(meta["chunking_fingerprint"])
        self.assertIn(meta["index_type"], {"faiss-flat-ip", "numpy-exact-ip"})
        self.assertEqual(meta["vectors"], meta["build_stats"]["chunks"])

    def test_search_hits_map_back_to_chunks(self):
        index = dense.DenseIndex(self.index_dir).load()
        hits = index.search_hits("Halden mill wheel Ostra", 5)
        self.assertTrue(hits)
        self.assertEqual([h["rank"] for h in hits], list(range(1, len(hits) + 1)))
        ids = json.loads((self.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        for hit in hits:
            self.assertIn(hit["chunk_id"], ids)
            self.assertEqual(hit["source"], "dense")
            self.assertLessEqual(hit["start"], hit["end"])
        self.assertIn("VB-1", [h["doc_id"] for h in hits])

    def test_reload_is_identical_and_documents_aggregate_once(self):
        first = dense.DenseRetriever(self.index_dir, {})
        second = dense.DenseRetriever(self.index_dir, {})
        a = first.search("flood of 1903 Ostra bridges", 10)
        b = second.search("flood of 1903 Ostra bridges", 10)
        self.assertEqual([(h["chunk_id"], round(h["score"], 6)) for h in a],
                         [(h["chunk_id"], round(h["score"], 6)) for h in b])
        docs = aggregate_documents(a, top_k=10)
        self.assertEqual(len({doc["doc_id"] for doc in docs}), len(docs))

    def test_empty_query_returns_no_candidates(self):
        self.assertEqual(dense.DenseIndex(self.index_dir).load().search_hits("   ", 5), [])

    def test_missing_index_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            dense.DenseIndex(Path(self.tmp.name) / "absent").load()

    def test_dimension_mismatch_is_detected(self):
        copy_dir = Path(self.tmp.name) / "mismatch"
        copy_dir.mkdir()
        for name in (dense.VECTORS_FILE, dense.SPANS_FILE, dense.CHUNK_IDS_FILE):
            (copy_dir / name).write_bytes((self.index_dir / name).read_bytes())
        meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        meta["dimension"] = 64
        (copy_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(Exception):
            dense.DenseIndex(copy_dir).load()

    def test_chunk_fingerprint_consistency_helper(self):
        report = dense.check_chunk_consistency(self.index_dir, self.chunks_dir)
        self.assertTrue(report["match"])


if __name__ == "__main__":
    unittest.main()
