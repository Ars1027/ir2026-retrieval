import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_baseline import prepare
from scripts.summarize_runs import summarize

class RunnerTests(unittest.TestCase):
    def test_config_environment_and_test_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config.json"
            config.write_text('{}')
            cmd, env = prepare("bm25", "train", config, root / "out", 3)
            self.assertEqual(env["IR2026_METHOD_CONFIG"], str(config.resolve()))
            self.assertNotIn("--config", cmd)
            with self.assertRaises(ValueError):
                prepare("bm25", "test", config, root / "out")
            (root / "out").mkdir()
            (root / "out" / "existing").touch()
            with self.assertRaises(ValueError):
                prepare("bm25", "dev", config, root / "out")

    def test_summary_keeps_unknown_tokens_and_excludes_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run_info.json").write_text(json.dumps({"split": "dev"}))
            rows = [{"id": "private", "answer": "secret", "status": "error", "usage": {"elapsed_seconds": 2, "model_input_tokens": None, "model_output_tokens": 0}}]
            (root / "results.jsonl").write_text(json.dumps(rows[0]))
            result = summarize(root)
            self.assertEqual(result["status_counts"], {"error": 1})
            self.assertIsNone(result["model_input_tokens"])
            self.assertEqual(result["p95_seconds"], 2)
            self.assertNotIn("secret", json.dumps(result))
            self.assertIsNone(result["gold_hit_at_10"])
