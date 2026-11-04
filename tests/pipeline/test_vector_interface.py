"""Reserved vector interface and baseline-identity tests (pipeline).

Run from the project root:
    python -m unittest discover -s tests/pipeline -v
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fixtures  # noqa: E402
from agent import bm25_solution, solution  # noqa: E402
from ir2026 import POLICY, CorpusTools  # noqa: E402
from pipeline import vector_stage  # noqa: E402
from pipeline.settings import ENV_VAR  # noqa: E402
from retrieval import contracts, interfaces  # noqa: E402

PLUGIN_SOURCE = '''
from retrieval import contracts, interfaces


class StubRetriever(interfaces.VectorRetriever):
    name = "stub-vector"

    def search(self, query, top_k):
        return [{"doc_id": "VB-2", "rank": 1, "score": 9.0},
                {"doc_id": "VB-1", "rank": 2, "score": 1.0}][:top_k]

    def describe(self):
        return {"name": self.name, "model": "stub", "revision": "test"}


class VectorFirstFusion(interfaces.DocumentFusion):
    name = "vector-first"

    def fuse(self, bm25_docs, vector_docs, top_k):
        by_id = {doc["doc_id"]: doc for doc in bm25_docs}
        ordered = []
        for hit in vector_docs:
            doc = by_id.get(hit["doc_id"])
            if doc is not None:
                ordered.append(doc)
        ordered.extend(doc for doc in bm25_docs if doc["doc_id"] not in
                       {d["doc_id"] for d in ordered})
        return [contracts.make_ranked_doc(doc["doc_id"], rank, doc["score"], doc["hits"])
                for rank, doc in enumerate(ordered[:top_k], 1)]


class BrokenRetriever(interfaces.VectorRetriever):
    name = "broken"

    def search(self, query, top_k):
        return [{"doc_id": "VB-1", "rank": 2, "score": 1.0}]


def create_retriever(settings):
    return StubRetriever()


def create_fusion(settings):
    return VectorFirstFusion()


def create_broken(settings):
    return BrokenRetriever()
'''

TEXTS = fixtures.DOCS


def make_tools(trace_dir=None):
    return CorpusTools(fixtures.FakeCorpus(TEXTS), "q1", dict(POLICY),
                       method_type="retrieval", trace_path=trace_dir)


def base_config(chunks_dir, bm25_dir, vector_block):
    return {"method": "test", "chunks_dir": str(chunks_dir),
            "bm25_index_dir": str(bm25_dir), "candidate_k": 50,
            "chunking": {"size": 200, "overlap": 40, "min_chars": 20},
            "bm25": {"k1": 0.9, "b": 0.4, "use_stopwords": False},
            "vector": vector_block,
            "retrieval": {"final_docs": 10, "evidence_docs": 5},
            "answer": {"mode": "extractive", "max_answer_chars": 200}}


class InterfaceSpecTests(unittest.TestCase):
    def test_descriptor_documents_input_output(self):
        spec = interfaces.interface_descriptor()
        self.assertEqual(spec["version"], interfaces.INTERFACE_VERSION)
        self.assertIn("search(query: str, top_k: int)", spec["vector_retriever"]["method"])
        self.assertEqual(tuple(spec["vector_retriever"]["hit_fields"]),
                         interfaces.VECTOR_HIT_FIELDS)
        self.assertIn("fuse(bm25_docs, vector_docs, top_k)",
                      spec["document_fusion"]["method"])

    def test_null_backend_returns_nothing(self):
        retriever = interfaces.NullVectorRetriever()
        self.assertEqual(retriever.search("anything", 10), [])
        self.assertFalse(retriever.describe()["enabled"])

    def test_validate_accepts_well_formed_hits(self):
        hits = fixtures.vector_hits(["VB-1", "VB-2"])
        self.assertEqual(interfaces.validate_vector_hits(hits, 5), [])

    def test_validate_rejects_malformed_output(self):
        good = fixtures.vector_hits(["VB-1", "VB-2"])
        cases = {
            "too many": fixtures.vector_hits(["VB-1", "VB-2"]) + [
                {"doc_id": "VB-3", "rank": 3, "score": 1.0}],
            "duplicate": [good[0], {"doc_id": "VB-1", "rank": 2, "score": 1.0}],
            "rank gap": [{"doc_id": "VB-1", "rank": 2, "score": 1.0}],
            "missing field": [{"doc_id": "VB-1", "rank": 1}],
            "non finite": [{"doc_id": "VB-1", "rank": 1, "score": float("nan")}],
            "not a list": {"doc_id": "VB-1"},
        }
        for label, payload in cases.items():
            with self.subTest(label):
                self.assertTrue(interfaces.validate_vector_hits(payload, 2))

    def test_disabled_config_builds_null_backend_and_fusion(self):
        settings = base_config("c", "b", {"enabled": False})
        self.assertIsInstance(interfaces.build_vector_retriever(settings),
                              interfaces.NullVectorRetriever)
        self.assertIsInstance(interfaces.build_fusion(settings), interfaces.NullFusion)

    def test_enabled_without_retriever_is_an_explicit_error(self):
        settings = base_config("c", "b", {"enabled": True, "retriever": None})
        with self.assertRaises(interfaces.RetrievalBackendError) as ctx:
            interfaces.build_vector_retriever(settings)
        self.assertIn("vector.retriever", str(ctx.exception))

    def test_missing_plugin_module_is_an_explicit_error(self):
        settings = base_config("c", "b", {"enabled": True,
                                          "retriever": "no_such_backend:create"})
        with self.assertRaises(interfaces.RetrievalBackendError) as ctx:
            interfaces.build_vector_retriever(settings)
        self.assertIn("cannot import", str(ctx.exception))

    def test_null_fusion_passes_bm25_ranking_through(self):
        docs = [contracts.make_ranked_doc("a", 1, 2.0, [{"chunk_id": "a#00000"}])]
        fused = interfaces.NullFusion().fuse(docs, fixtures.vector_hits(["b"]), 10)
        self.assertEqual([d["doc_id"] for d in fused], ["a"])


class BaselineIdentityTests(unittest.TestCase):
    """The shipped baseline must not change when the hook is disabled."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        work = Path(cls.tmp.name)
        cls.corpus_root = fixtures.write_corpus(work / "corpus")
        cls.bm25_dir = fixtures.build_bm25_index(cls.corpus_root, work / "idx")
        cls.chunks_dir = work / "idx" / "chunks"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run_with_config(self, config, agent_module, trace_dir=None):
        path = Path(self.tmp.name) / f"config_{abs(hash(json.dumps(config, sort_keys=True)))}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        previous = os.environ.get(ENV_VAR)
        os.environ[ENV_VAR] = str(path)
        try:
            tools = make_tools(trace_dir)
            result = agent_module.solve({"id": "q1", "question": "Halden mill wheel Ostra"},
                                        tools, {"method_type": "retrieval"})
            return result, tools
        finally:
            if previous is None:
                os.environ.pop(ENV_VAR, None)
            else:
                os.environ[ENV_VAR] = previous

    def test_disabled_hook_matches_bm25_adapter_exactly(self):
        config = base_config(self.chunks_dir, self.bm25_dir, {"enabled": False})
        baseline, baseline_tools = self._run_with_config(config, bm25_solution, None)
        default, default_tools = self._run_with_config(config, solution, None)
        for key in ("status", "answer", "doc_ids", "evidence"):
            self.assertEqual(default[key], baseline[key], key)
        self.assertEqual([event["tool"] for event in default_tools.trace],
                         [event["tool"] for event in baseline_tools.trace])
        self.assertEqual([event["tool"] for event in default_tools.trace],
                         ["load_index", "bm25_search", "document_aggregate",
                          "read_evidence", "answer"])

    def test_disabled_hook_keeps_ranked_documents(self):
        config = base_config(self.chunks_dir, self.bm25_dir, {"enabled": False})
        result, tools = self._run_with_config(config, solution, None)
        self.assertIn(result["status"], {"ok", "no_answer"})
        self.assertTrue(result["doc_ids"])
        for quote in result["evidence"]:
            self.assertIn(quote["text"], TEXTS[quote["doc_id"]])
        self.assertEqual(result["usage"]["tool_calls"], 0)


class PluginIntegrationTests(unittest.TestCase):
    """An external backend can be plugged in without touching the baseline."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        work = Path(cls.tmp.name)
        cls.corpus_root = fixtures.write_corpus(work / "corpus")
        cls.bm25_dir = fixtures.build_bm25_index(cls.corpus_root, work / "idx")
        cls.chunks_dir = work / "idx" / "chunks"
        cls.plugin_dir = work / "plugins"
        cls.plugin_dir.mkdir()
        (cls.plugin_dir / "stub_vector_plugin.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
        sys.path.insert(0, str(cls.plugin_dir))
        importlib.invalidate_caches()
        sys.modules.pop("stub_vector_plugin", None)

    @classmethod
    def tearDownClass(cls):
        if str(cls.plugin_dir) in sys.path:
            sys.path.remove(str(cls.plugin_dir))
        sys.modules.pop("stub_vector_plugin", None)
        cls.tmp.cleanup()

    def _settings(self, retriever, fusion):
        return base_config(self.chunks_dir, self.bm25_dir,
                           {"enabled": True, "retriever": retriever,
                            "fusion": fusion, "top_k": 10,
                            "index_dir": str(Path(self.tmp.name) / "vector")})

    def test_plugin_ranking_reaches_the_assembler(self):
        settings = self._settings("stub_vector_plugin:create_retriever",
                                  "stub_vector_plugin:create_fusion")
        tools = make_tools(None)
        result = vector_stage.solve({"id": "q1", "question": "Halden mill wheel Ostra"},
                                    tools, {"method_type": "retrieval"}, settings)
        stages = [event["tool"] for event in tools.trace]
        self.assertEqual(stages, ["load_indexes", "bm25_search", "document_aggregate",
                                  "vector_search", "fuse", "read_evidence", "answer"])
        self.assertEqual(result["doc_ids"][0], "VB-2")
        self.assertEqual(interfaces.validate_ranked_docs(
            [dict(doc_id=doc_id, rank=rank, score=1.0, hits=[{}])
             for rank, doc_id in enumerate(result["doc_ids"], 1)]), [])

    def test_dense_only_skips_bm25_and_preserves_vector_evidence(self):
        settings = self._settings("stub_vector_plugin:create_retriever",
                                  "stub_vector_plugin:create_fusion")
        settings["vector"]["dense_only"] = True
        text = TEXTS["VB-2"]
        chunk = contracts.make_chunk("VB-2", 0, 0, len(text), text)
        hit = contracts.make_hit(chunk, 0.9, 1, "dense")
        document = contracts.make_ranked_doc("VB-2", 1, 0.9, [hit])

        class DenseFixture(interfaces.VectorRetriever):
            last_query_tokens = 7
            def search(self, query, top_k):
                return [document]

        class DenseFusion(interfaces.DocumentFusion):
            def fuse(inner_self, bm25_docs, vector_docs, top_k):
                self.assertEqual(bm25_docs, [])
                return vector_docs[:top_k]

        tools = make_tools(None)
        with patch.object(vector_stage, "_bm25", side_effect=AssertionError("BM25 must not load")) as bm25, \
             patch.object(interfaces, "build_vector_retriever", return_value=DenseFixture()), \
             patch.object(interfaces, "build_fusion", return_value=DenseFusion()):
            result = vector_stage.solve({"id": "q1", "question": "When did Ostra museum open?"},
                                        tools, {"method_type": "retrieval"}, settings)
        bm25.assert_not_called()
        stages = [event["tool"] for event in tools.trace]
        self.assertEqual(stages, ["load_indexes", "vector_search", "fuse", "read_evidence", "answer"])
        self.assertEqual(result["doc_ids"], ["VB-2"])
        self.assertTrue(result["evidence"])
        self.assertTrue(all(quote["text"] in text for quote in result["evidence"]))
        self.assertEqual(result["usage"]["model_input_tokens"], 7)

    def test_broken_plugin_is_explicit_failure(self):
        settings = self._settings("stub_vector_plugin:create_broken",
                                  "stub_vector_plugin:create_fusion")
        tools = make_tools(None)
        with self.assertRaises(interfaces.RetrievalBackendError):
            vector_stage.solve({"id": "q1", "question": "Halden mill wheel Ostra"},
                               tools, {"method_type": "retrieval"}, settings)
        vector_event = [event for event in tools.trace if event["tool"] == "vector_search"][0]
        self.assertTrue(vector_event["output"]["errors"])

    def test_fusion_output_is_validated(self):
        settings = self._settings("stub_vector_plugin:create_retriever", None)
        tools = make_tools(None)
        result = vector_stage.solve({"id": "q1", "question": "Halden mill wheel Ostra"},
                                    tools, {"method_type": "retrieval"}, settings)
        # No fusion configured: the BM25 ranking is kept unchanged.
        self.assertTrue(result["doc_ids"])
        self.assertEqual(len(result["doc_ids"]), len(set(result["doc_ids"])))


if __name__ == "__main__":
    unittest.main()
