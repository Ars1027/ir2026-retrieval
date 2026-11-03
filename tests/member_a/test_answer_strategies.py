"""Answer strategy tests: typed span selection versus the sentence fallback."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fixtures  # noqa: E402
from ir2026 import CorpusTools, POLICY  # noqa: E402
from pipeline import answer, evidence  # noqa: E402
from pipeline.result_builder import assemble_result  # noqa: E402
from retrieval import contracts  # noqa: E402

TEXT = fixtures.DOCS["ZA-1"]
QUESTION = "What is the full name of the person who founded the Zarnia Observatory in 1907?"


def ranked_doc(doc_id="ZA-1", score=9.0):
    text = fixtures.DOCS[doc_id]
    chunk = {"chunk_id": f"{doc_id}#00000", "doc_id": doc_id, "start": 0,
             "end": len(text), "text": text}
    hit = contracts.make_hit(chunk, score, 1, "bm25")
    return contracts.make_ranked_doc(doc_id, 1, score, [hit])


def tools_for(qid="q1"):
    return CorpusTools(fixtures.FakeCorpus(fixtures.DOCS), qid, dict(POLICY),
                       method_type="retrieval")


class AnswerTypeTests(unittest.TestCase):
    def test_type_inference(self):
        self.assertEqual(answer.expected_type("How many visitors came in 1990?"), "quantity")
        self.assertEqual(answer.expected_type("What is the title of the book by Oda Vell?"), "title")
        self.assertEqual(answer.expected_type(QUESTION), "person")
        self.assertEqual(answer.expected_type("Which company manufactures the brass lens?"), "org")
        self.assertEqual(answer.expected_type("In what year was the dock completed?"), "date")

    def test_candidate_spans_cover_entities_and_quantities(self):
        text = ("The Kestrel Ridge archive holds the 1788 customs ledger and 42 photographs. "
                "A report titled \"Brass and Wind\" was published in 2003 with a 12% rise. "
                "and Apple released nothing here.")
        spans = answer.candidate_spans(text)
        texts = {span["text"] for span in spans}
        kinds = {(span["text"], span["kind"]) for span in spans}
        self.assertIn(("42", "quantity") or ("42", "date"), kinds)
        self.assertTrue(any("Brass and Wind" in item for item in texts))
        self.assertTrue(any("2003" in item for item in texts))
        self.assertTrue(any("12%" in item for item in texts))
        self.assertFalse(any(item.startswith("and ") for item in texts))


class SpanSelectionTests(unittest.TestCase):
    def test_selects_person_name_from_top_document(self):
        selection = answer.select_answer(QUESTION, [ranked_doc()], fixtures.FakeCorpus(fixtures.DOCS).text)
        self.assertEqual(selection["answer"], "Ilse Marrow")
        self.assertEqual(selection["kind"], "person")
        self.assertEqual(selection["doc_id"], "ZA-1")

    def test_selected_span_lies_inside_its_quote_window(self):
        corpus = fixtures.FakeCorpus(fixtures.DOCS)
        selection = answer.select_answer(QUESTION, [ranked_doc()], corpus.text)
        start, end = selection["quote"]
        quoted = TEXT[start:end]
        self.assertIn(selection["answer"], quoted)
        self.assertGreaterEqual(len(quoted), POLICY["min_quote_chars"])

    def test_no_keyword_support_returns_empty_selection(self):
        selection = answer.select_answer("What is the name of the ship in the Antarctic fleet?",
                                         [ranked_doc()], fixtures.FakeCorpus(fixtures.DOCS).text)
        self.assertEqual(selection, {})

    def test_window_around_keeps_the_span(self):
        start = TEXT.index("Ilse Marrow")
        win_start, win_end = evidence.window_around(TEXT, start, start + 11, 20, 120)
        self.assertIn("Ilse Marrow", TEXT[win_start:win_end])
        self.assertLessEqual(win_end - win_start, 120)


class AssemblyTests(unittest.TestCase):
    def _settings(self, span_selection):
        config = {"answer": {"mode": "extractive", "max_answer_chars": 200},
                  "retrieval": {"final_docs": 10, "evidence_docs": 5}}
        if span_selection is not None:
            config["answer"]["span_selection"] = span_selection
        return config

    def test_span_mode_answers_with_the_entity_and_anchors_the_quote(self):
        result = assemble_result({"id": "q1", "question": QUESTION}, [ranked_doc()],
                                 tools_for(), self._settings(True), source="bm25")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], "Ilse Marrow")
        self.assertEqual(result["doc_ids"], ["ZA-1"])
        self.assertTrue(result["evidence"])
        self.assertTrue(any(result["answer"] in item["text"] for item in result["evidence"]))
        for item in result["evidence"]:
            self.assertIn(item["text"], fixtures.DOCS[item["doc_id"]])

    def test_default_mode_keeps_the_sentence_answer(self):
        result = assemble_result({"id": "q1", "question": QUESTION}, [ranked_doc()],
                                 tools_for(), self._settings(None), source="bm25")
        self.assertEqual(result["status"], "ok")
        self.assertNotEqual(result["answer"], "Ilse Marrow")
        self.assertGreater(len(result["answer"]), 40)
        self.assertIn(result["answer"], TEXT)

    def test_span_answer_never_exceeds_the_limit(self):
        settings = self._settings(True)
        settings["answer"]["max_answer_chars"] = 8
        result = assemble_result({"id": "q1", "question": QUESTION}, [ranked_doc()],
                                 tools_for(), settings, source="bm25")
        self.assertLessEqual(len(result["answer"]), 8)


if __name__ == "__main__":
    unittest.main()
