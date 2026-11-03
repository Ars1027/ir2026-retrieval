"""Contract and configuration tests (member A).

Run from the project root:
    python -m unittest discover -s tests/member_a -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import settings as settings_mod  # noqa: E402
from retrieval import contracts  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_chunk_id_roundtrip_and_stability(self):
        first = contracts.chunk_id("ZA-1", 7)
        second = contracts.chunk_id("ZA-1", 7)
        self.assertEqual(first, second)
        self.assertEqual(contracts.split_chunk_id(first), ("ZA-1", 7))
        self.assertNotEqual(contracts.chunk_id("ZA-1", 7), contracts.chunk_id("ZA-1", 8))

    def test_invalid_chunk_ids_rejected(self):
        with self.assertRaises(contracts.ContractError):
            contracts.chunk_id("", 0)
        with self.assertRaises(contracts.ContractError):
            contracts.split_chunk_id("broken-id")

    def test_chunk_validation_checks_offsets_and_text(self):
        text = "The archive keeps the 1788 customs ledger in Belport."
        chunk = contracts.make_chunk("ZA-4", 0, 4, 20, text[4:20])
        self.assertEqual(contracts.validate_chunk(chunk, text), [])
        bad = dict(chunk, text="rewritten text")
        self.assertTrue(any("source slice" in problem
                            for problem in contracts.validate_chunk(bad, text)))

    def test_ranked_doc_validation(self):
        hit = {"chunk_id": "ZA-1#00000", "doc_id": "ZA-1", "start": 0, "end": 10,
               "score": 1.5, "rank": 1, "source": "bm25"}
        docs = [contracts.make_ranked_doc("ZA-1", 1, 1.5, [hit])]
        self.assertEqual(contracts.validate_ranked_docs(docs), [])
        broken = [contracts.make_ranked_doc("ZA-1", 2, 1.5, [hit])]
        self.assertTrue(contracts.validate_ranked_docs(broken))

    def test_chunking_config_defaults_and_guards(self):
        resolved = contracts.chunking_config({})
        self.assertEqual(resolved["size"], contracts.DEFAULT_CHUNKING["size"])
        with self.assertRaises(contracts.ContractError):
            contracts.chunking_config({"chunking": {"size": 50, "min_chars": 80}})


class SettingsTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(settings_mod.ENV_VAR, None)

    def test_default_config_is_bm25(self):
        os.environ.pop(settings_mod.ENV_VAR, None)
        loaded = settings_mod.load_settings()
        self.assertTrue(loaded["config_path"].endswith("bm25.json"))
        self.assertTrue(Path(loaded["chunks_dir"]).is_absolute())

    def test_env_points_to_config_and_keeps_member_b_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hybrid.json"
            path.write_text(json.dumps({
                "method": "hybrid-test",
                "dense_index_dir": "artifacts/member_b/dense",
                "rrf_k": 42,
                "candidate_k": 50,
                "dense": {"encoder": {"model": "some/model", "revision": "abc"}},
            }), encoding="utf-8")
            os.environ[settings_mod.ENV_VAR] = str(path)
            loaded = settings_mod.load_settings()
        self.assertEqual(loaded["method"], "hybrid-test")
        self.assertEqual(loaded["rrf_k"], 42)
        self.assertEqual(loaded["dense"]["encoder"]["revision"], "abc")
        self.assertTrue(Path(loaded["dense_index_dir"]).is_absolute())
        self.assertEqual(loaded["answer"]["mode"], settings_mod.DEFAULTS["answer"]["mode"])

    def test_invalid_configs_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"candidate_k": 0}), encoding="utf-8")
            with self.assertRaises(settings_mod.SettingsError):
                settings_mod.load_settings(path)
            missing = Path(tmp) / "absent.json"
            with self.assertRaises(settings_mod.SettingsError):
                settings_mod.load_settings(missing)


if __name__ == "__main__":
    unittest.main()
