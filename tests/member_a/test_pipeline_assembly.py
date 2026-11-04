"""Evidence, answer and result-assembly tests (member A)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ir2026 import POLICY, CorpusTools  # noqa: E402
from pipeline import evidence  # noqa: E402
from pipeline.answer import extractive_answer, make_answer  # noqa: E402
from pipeline.result_builder import assemble_result  # noqa: E402
import fixtures  # noqa: E402

SETTINGS = {
    "retrieval": {"final_docs": 10, "evidence_docs": 5},
    "answer": {"mode": "extractive", "max_answer_chars": 200},
}


def tools_for(texts, qid="q1"):
    return CorpusTools(fixtures.FakeCorpus(texts), qid, dict(POLICY), method_type="retrieval")


def ranked(doc_hits):
    from retrieval.document_rank import aggregate_documents
    hits = []
    rank = 0
    for doc_id, start, end in doc_hits:
        rank += 1
        hits.append({"chunk_id": f"{doc_id}#{rank:05d}", "doc_id": doc_id, "start": start,
                     "end": end, "score": 10.0 - rank, "rank": rank, "source": "bm25"})
    return aggregate_documents(hits, top_k=10)


class EvidenceTests(unittest.TestCase):
    def test_quote_is_verbatim_and_within_budget(self):
        text = "The Kestrel Ridge archive holds the 1788 customs ledger. " * 30
        start, end = evidence.clip_span(text, 10, 40, 20, 300)
        quote = text[start:end]
        self.assertTrue(20 <= len(quote) <= 300)
        self.assertIn(quote, text)

    def test_long_span_is_trimmed_at_a_sentence_break(self):
        text = ("Sentence number one is here. " * 60)
        start, end = evidence.clip_span(text, 0, 900, 20, 200)
        self.assertLessEqual(end - start, 200)
        self.assertTrue(text[start:end] in text)

    def test_short_span_is_extended_to_the_minimum(self):
        text = "Belport trading port 1712 dry dock customs house ledger archive survey."
        start, end = evidence.clip_span(text, 0, 5, 20, 1200)
        self.assertGreaterEqual(end - start, 20)

    def test_quotes_are_limited_and_unique(self):
        texts = {"d1": "Alpha sentence about the ledger archive in Belport. " * 20,
                 "d2": "Beta sentence about the Kestrel Ridge plateau survey. " * 20}
        docs = ranked([("d1", 0, 60), ("d2", 0, 60)])
        quotes = evidence.build_quotes(docs, lambda d: texts[d], dict(POLICY))
        self.assertLessEqual(len(quotes), POLICY["max_evidence_quotes"])
        self.assertEqual(len({q["text"] for q in quotes}), len(quotes))


class AssemblyTests(unittest.TestCase):
    TEXT = ("The Zarnia Observatory was founded in 1907 by Ilse Marrow near the town of Belport. "
            "Its first director catalogued the Pelican Nebula using a brass refractor. " * 4)

    def test_recorder_failure_propagates(self):
        from unittest.mock import patch
        tools = tools_for({"ZA-1": self.TEXT})
        with patch.object(tools, "record_retrieval", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                assemble_result({"id":"q1", "question":"observatory"}, ranked([("ZA-1",0,100)]), tools, SETTINGS)

    def test_answer_elapsed_includes_call(self):
        from unittest.mock import patch
        from pipeline import result_builder
        tools = tools_for({"ZA-1": self.TEXT})
        ticks = [0.0]
        def answer(*args, **kwargs):
            ticks[0] += 2.0
            return {"answer":"1907", "usage":{}}
        with patch.object(result_builder.time, "monotonic", side_effect=lambda: ticks[0]), patch.object(result_builder.answer_mod, "make_answer", side_effect=answer):
            assemble_result({"id":"q1", "question":"observatory"}, ranked([("ZA-1",0,100)]), tools, SETTINGS)
        event = next(e for e in tools.trace if e["tool"] == "answer")
        self.assertEqual(event["elapsed_ms"], 2000.0)

    def test_ok_result_has_short_answer_and_valid_quotes(self):
        tools = tools_for({"ZA-1": self.TEXT})
        docs = ranked([("ZA-1", 0, 120)])
        result = assemble_result({"id": "q1", "question": "Who founded the Zarnia Observatory?"},
                                 docs, tools, SETTINGS)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["answer"].strip())
        self.assertLess(len(result["answer"]), 200)
        self.assertNotEqual(result["answer"], self.TEXT)
        self.assertEqual(result["doc_ids"], ["ZA-1"])
        self.assertTrue(1 <= len(result["evidence"]) <= 5)
        for quote in result["evidence"]:
            self.assertIn(quote["doc_id"], result["doc_ids"])
            self.assertTrue(20 <= len(quote["text"]) <= 1200)
            self.assertIn(quote["text"], self.TEXT)
        self.assertEqual(result["usage"]["model_input_tokens"], 0)
        self.assertEqual(result["usage"]["tool_calls"], 0)
        self.assertTrue(result["usage"]["token_notes"])

    def test_no_candidates_returns_no_answer(self):
        tools = tools_for({"ZA-1": self.TEXT})
        result = assemble_result({"id": "q1", "question": "Anything?"}, [], tools, SETTINGS)
        self.assertEqual(result["status"], "no_answer")
        self.assertEqual((result["doc_ids"], result["evidence"]), ([], []))

    def test_insufficient_evidence_keeps_documents(self):
        tools = tools_for({"ZA-1": self.TEXT})
        docs = ranked([("ZA-1", 0, 60)])
        result = assemble_result({"id": "q2", "question": "quantum chromodynamics lattice spacing"},
                                 docs, tools, SETTINGS)
        self.assertEqual(result["status"], "no_answer")
        self.assertEqual(result["doc_ids"], ["ZA-1"])
        self.assertTrue(result["evidence"])

    def test_documents_without_usable_quote_are_dropped(self):
        tools = tools_for({"tiny": "Too short."})
        docs = ranked([("tiny", 0, 5)])
        result = assemble_result({"id": "q3", "question": "What is short?"}, docs, tools, SETTINGS)
        self.assertEqual(result["doc_ids"], [])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["status"], "no_answer")

    def test_model_mode_without_key_degrades_honestly(self):
        os.environ.pop("IR2026_MODEL_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        tools = tools_for({"ZA-1": self.TEXT})
        docs = ranked([("ZA-1", 0, 120)])
        settings = {"retrieval": SETTINGS["retrieval"],
                    "answer": {"mode": "model", "model": "unused/model",
                               "api_key_env": "IR2026_MODEL_API_KEY"}}
        result = assemble_result({"id": "q4", "question": "Who founded the observatory?"},
                                 docs, tools, settings)
        self.assertEqual(result["status"], "no_answer")
        self.assertTrue(result["doc_ids"])
        self.assertTrue(result["evidence"])
        self.assertIn("error", result)
        self.assertIsNone(result["usage"]["model_input_tokens"])
        self.assertTrue(result["usage"]["token_notes"])

    def test_answer_disabled_uses_zero_tokens(self):
        payload = make_answer("Who founded it?", [{"doc_id": "d1", "text": "x" * 40}],
                              {"answer": {"mode": "off"}})
        self.assertEqual(payload["usage"]["model_input_tokens"], 0)
        self.assertEqual(payload["answer"], "")

    def test_extractive_answer_rejects_number_dumps(self):
        quote = {"doc_id": "d1", "text": "1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002."}
        answer, _ = extractive_answer("Which years were listed in the report?", [quote])
        self.assertEqual(answer, "")

    def test_assembler_logs_stages_for_retrieval_pipeline(self):
        tools = tools_for({"ZA-1": self.TEXT})
        docs = ranked([("ZA-1", 0, 120)])
        assemble_result({"id": "q5", "question": "Who founded the Zarnia Observatory?"},
                        docs, tools, SETTINGS)
        stages = [event["tool"] for event in tools.trace]
        self.assertEqual(stages, ["read_evidence", "answer"])
        self.assertTrue(all(event["event_type"] == "retrieval" for event in tools.trace))


if __name__ == "__main__":
    unittest.main()
