"""Synthetic integration and leakage/budget guards; no benchmark QA is included."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ir2026 import CorpusTools, POLICY, validate_results, write_jsonl
from posttraining.agent import run_policy
from posttraining.config import comparison_signature
from posttraining.data import build_dataset, training_ids
from posttraining.policy import SYSTEM_PROMPT, encode, messages_for, parse_action
from posttraining.tools import IndexedTools


class Corpus:
    def __init__(self):
        self.docs = {"fiction-1": {}, "fiction-2": {}}
        self.texts = {"fiction-1": "海湾档案馆 opened in 1894. Its curator was Mara Venn. " * 8,
                      "fiction-2": "The lighthouse catalogue was revised in 1912. " * 8}

    def text(self, doc_id):
        return self.texts[doc_id]


class Retriever:
    def search(self, query, top_k):
        return [{"doc_id": "fiction-1", "chunk_id": "fiction-1#00000", "start": 0,
                 "end": 300, "rank": 1, "score": 3.0, "source": "bm25"}]


class Client:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []

    def complete(self, messages, deadline):
        self.inputs.append(messages)
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value, {"input": 10, "output": 5}


class Tokenizer:
    """Prefix-preserving fictional chat template, used without model dependencies."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>" + m["content"] + "</turn>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode())


class PolicyTests(unittest.TestCase):
    def tools(self):
        supplied = CorpusTools(Corpus(), "fiction-q", method_type="agent")
        return supplied, IndexedTools(supplied, Retriever(), {})

    def test_rejects_unrecognized_fields_and_non_json(self):
        for raw in ['hello', '{"action":"delete"}', '{"action":"search","query":"hi","extra":1}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_action(raw)

    def test_rejects_unsafe_read_and_duplicate_finish(self):
        for obj in [{"action": "read", "doc_id": "fiction-1", "start": True},
                    {"action": "read", "doc_id": "fiction-1", "start": -1},
                    {"action": "finish", "doc_ids": ["x", "x"], "answer": ""}]:
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                parse_action(encode(obj))

    def test_final_result_object_is_a_valid_terminal_action(self):
        action = parse_action('{"doc_ids":["fiction-1"],"answer":"Mara Venn"}')
        self.assertEqual(action["action"], "finish")
        with self.assertRaises(ValueError):
            parse_action('{"answer":"unsupported prior knowledge"}')

    def test_history_keeps_complete_recent_pairs(self):
        pair = [{"role": "assistant", "content": "old"}, {"role": "user", "content": "observation"}]
        latest = [{"role": "assistant", "content": "new"}, {"role": "user", "content": "result"}]
        messages = messages_for("fiction?", [pair, latest], history_chars=9)
        self.assertEqual(messages[2:], latest)
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)

    def test_end_to_end_trace_quotes_and_tokens_pass_official_validator(self):
        supplied, tools = self.tools()
        client = Client([encode({"action": "search", "query": "bay archive"}),
                         encode({"action": "read", "doc_id": "fiction-1", "start": 0}),
                         encode({"action": "finish", "doc_ids": ["fiction-1"], "answer": "Mara Venn"})])
        result = run_policy({"id": "fiction-q", "question": "Who was the curator?"}, tools, client, {})
        self.assertEqual(result["usage"]["tool_calls"], 2)
        self.assertEqual(result["usage"]["model_input_tokens"], 30)
        self.assertEqual(result["usage"]["model_calls"], 3)
        self.assertIs(supplied.trace, tools.trace)
        self.assertIn(result["evidence"][0]["text"], tools.corpus.text("fiction-1"))
        with tempfile.TemporaryDirectory() as tmp:
            write_jsonl(Path(tmp) / "fiction-q.jsonl", tools.trace)
            errors = validate_results([{"id": "fiction-q", "question": "Who was the curator?"}],
                                      [result], tools.corpus, POLICY, Path(tmp), method_type="agent")
        self.assertEqual(errors, [])

    def test_invalid_model_action_is_repaired_without_inventing_a_tool_call(self):
        _, tools = self.tools()
        client = Client(['invalid', encode({"action": "finish", "doc_ids": [], "answer": ""})])
        result = run_policy({"id": "fiction-q", "question": "Fiction?"}, tools, client, {})
        self.assertEqual(result["usage"]["invalid_actions"], 1)
        self.assertEqual(result["usage"]["tool_calls"], 0)
        self.assertTrue(client.inputs[1][-1]["content"].startswith("ACTION_ERROR"))

    def test_unseen_ids_cannot_be_read_or_returned(self):
        _, tools = self.tools()
        with self.assertRaises(ValueError):
            tools.read_observed("fiction-2", 0)
        with self.assertRaises(ValueError):
            tools.finish({"doc_ids": ["fiction-2"], "answer": "made up"})

    def test_optional_read_end_uses_unicode_offsets_and_keeps_the_budget(self):
        _, tools = self.tools()
        tools.search("archive")
        action = parse_action(encode({"action": "read", "doc_id": "fiction-1", "start": 1, "end": 31}))
        observation = tools.read_observed(action["doc_id"], action["start"], action["end"])
        self.assertEqual(observation["text"], tools.corpus.text("fiction-1")[1:31])
        self.assertEqual(tools.trace[-1]["arguments"]["max_chars"], 30)
        with self.assertRaises(ValueError):
            parse_action(encode({"action": "read", "doc_id": "fiction-1", "start": 31, "end": 1}))

    def test_call_budget_is_enforced_and_failure_stays_in_results(self):
        _, tools = self.tools()
        client = Client([encode({"action": "search", "query": "archive"})] * 3)
        result = run_policy({"id": "fiction-q", "question": "Fiction?"}, tools, client,
                            {"max_tool_calls": 2})
        self.assertEqual(result["status"], "budget_exceeded")
        self.assertEqual(result["usage"]["tool_calls"], 2)
        self.assertEqual(len(tools.trace), 2)

    def test_timeout_retains_an_empty_record(self):
        _, tools = self.tools()
        tools.started = time.monotonic() - 601
        result = run_policy({"id": "fiction-q", "question": "Fiction?"}, tools, Client([]), {})
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["doc_ids"], [])

    def test_failed_generation_does_not_report_zero_unknown_tokens(self):
        _, tools = self.tools()
        result = run_policy({"id": "fiction-q", "question": "Fiction?"}, tools,
                            Client([RuntimeError("generation failed")]), {})
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["usage"]["model_input_tokens"])

    def test_comparison_requires_same_model_and_retrieval(self):
        before = {"method": "before", "model": {"id": "fiction-model", "revision": "v1", "adapter_path": None},
                  "policy": {"max_tool_calls": 10}}
        after = {**before, "method": "after", "model": {**before["model"], "adapter_path": "adapter"}}
        self.assertEqual(comparison_signature(before), comparison_signature(after))
        after["model"]["revision"] = "v2"
        self.assertNotEqual(comparison_signature(before), comparison_signature(after))


class DataTests(unittest.TestCase):
    def fixture(self, root):
        data, run = root / "datas", root / "run"
        train = [{"id": f"fiction-q{i}", "question": f"Who curated fictional archive number {i}?"} for i in range(20)]
        write_jsonl(data / "train.jsonl", train)
        for split in ("dev", "test"):
            write_jsonl(data / f"{split}.jsonl", [{"id": split, "question": split + " fictional question?"}])
        write_jsonl(data / "train_labels.jsonl", [{"id": q["id"], "answer": "Mara Venn",
                    "gold_doc_ids": ["fiction-1"]} for q in train])
        run.mkdir()
        (run / "run_info.json").write_text(json.dumps({"split": "train", "method_type": "agent"}))
        write_jsonl(run / "results.jsonl", [{"id": q["id"], "status": "ok", "doc_ids": ["fiction-1"]} for q in train])
        for q in train:
            messages = messages_for(q["question"], [])
            write_jsonl(run / "decisions" / f"{q['id']}.jsonl", [
                {"turn": 0, "action_valid": True, "messages": messages,
                 "output": encode({"action": "search", "query": "fictional archive"})},
                {"turn": 1, "action_valid": True, "messages": messages_for(q["question"], [[
                    {"role": "assistant", "content": encode({"action": "search", "query": "fictional archive"})},
                    {"role": "user", "content": "TOOL_RESULT Fictional archive: Mara Venn was the curator."}]]),
                 "output": encode({"action": "finish", "doc_ids": ["fiction-1"], "answer": "Mara Venn"})}])
        return data, run

    def test_question_group_split_and_loss_only_on_last_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, run = self.fixture(root)
            output = root / "sft"
            stats = build_dataset(data, run, output, Tokenizer())
            self.assertEqual(stats["examples"], {"train": 36, "valid": 4})
            self.assertFalse(stats["tool_observations_are_targets"])
            provenance = read_rows(output / "provenance.jsonl")
            train_ids = {r["question_id"] for r in provenance if r["split"] == "train"}
            valid_ids = {r["question_id"] for r in provenance if r["split"] == "valid"}
            self.assertFalse(train_ids & valid_ids)
            for sample in read_rows(output / "train.jsonl"):
                self.assertEqual(sample["messages"][-1]["role"], "assistant")
                full = Tokenizer().apply_chat_template(sample["messages"])
                offset = len(Tokenizer().apply_chat_template(sample["messages"][:-1], add_generation_prompt=True))
                self.assertNotIn("TOOL_RESULT", bytes(full[offset:]).decode())

    def test_rejects_dev_trajectories_and_incomplete_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, run = self.fixture(Path(tmp))
            (run / "run_info.json").write_text(json.dumps({"split": "dev", "method_type": "agent"}))
            with self.assertRaisesRegex(ValueError, "train"):
                build_dataset(data, run, Path(tmp) / "sft", Tokenizer())
            (run / "run_info.json").write_text(json.dumps({"split": "train", "method_type": "agent"}))
            write_jsonl(run / "results.jsonl", [])
            with self.assertRaisesRegex(ValueError, "complete"):
                build_dataset(data, run, Path(tmp) / "sft", Tokenizer())

    def test_preprocessing_never_requires_dev_or_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, run = self.fixture(Path(tmp))
            (data / "dev.jsonl").unlink()
            (data / "test.jsonl").unlink()
            stats = build_dataset(data, run, Path(tmp) / "sft", Tokenizer())
            self.assertEqual(stats["source_questions"], 20)
            self.assertEqual(set(stats["source_sha256"]), {"train.jsonl", "train_labels.jsonl"})

    def test_overlength_is_filtered_instead_of_silent_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, run = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "empty"):
                build_dataset(data, run, Path(tmp) / "sft", Tokenizer(), max_tokens=1)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


if __name__ == "__main__":
    unittest.main()
