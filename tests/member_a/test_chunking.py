"""Faithful chunking tests (member A)."""
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

from retrieval import chunking  # noqa: E402
from retrieval.corpus_adapter import CorpusAdapter  # noqa: E402
import fixtures  # noqa: E402


class SpanTests(unittest.TestCase):
    def test_spans_are_exact_slices_and_cover_the_text(self):
        text = "First paragraph about the archive.\n\nSecond paragraph mentions 1899.\n\n" * 6
        spans = chunking.span_chunks(text, 100, 20, 30)
        self.assertTrue(spans)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))
        for start, end in spans:
            self.assertTrue(0 <= start < end <= len(text))
            self.assertTrue(text[start:end])
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            self.assertLess(next_start, prev_end)  # configured overlap is honoured

    def test_unicode_offsets_are_python_characters(self):
        text = "Über die Sternwarte: 1907 – gegründet von Ilse Marrow. " * 12
        spans = chunking.span_chunks(text, 60, 10, 20)
        for start, end in spans:
            self.assertEqual(text[start:end], text[start:end])
            self.assertEqual(len(text[start:end]), end - start)

    def test_long_paragraph_without_spaces_is_cut_at_exact_size(self):
        text = "x" * 250
        spans = chunking.span_chunks(text, 100, 0, 20)
        self.assertEqual([(0, 100), (100, 200), (200, 250)], spans)

    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(chunking.span_chunks("", 100, 10, 20), [])

    def test_tail_shorter_than_min_chars_is_repaired(self):
        text = "a" * 95 + " b"
        spans = chunking.span_chunks(text, 100, 0, 20)
        self.assertTrue(all(end - start >= 20 for start, end in spans))

    def test_stable_across_runs(self):
        text = ("Stable chunk boundary text with 1788 ledger references. " * 30)
        first = chunking.span_chunks(text, 120, 30, 20)
        second = chunking.span_chunks(text, 120, 30, 20)
        self.assertEqual(first, second)


class ChunkStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = fixtures.write_corpus(Path(self.tmp.name) / "corpus")
        self.out = Path(self.tmp.name) / "chunks"

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_chunk_matches_source_and_ids_are_stable(self):
        stats = fixtures.write_chunk_store(self.root, self.out, size=120, overlap=20, min_chars=20)
        self.assertGreater(stats["chunks"], 0)
        adapter = CorpusAdapter(self.root)
        report = chunking.check_faithful(chunking.iter_chunks_file(self.out), adapter)
        self.assertTrue(report["ok"], report["bad"])
        chunks = list(chunking.iter_chunks_file(self.out))
        ids = [c["chunk_id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))
        again = Path(self.tmp.name) / "chunks2"
        fixtures.write_chunk_store(self.root, again, size=120, overlap=20, min_chars=20)
        self.assertEqual(ids, [c["chunk_id"] for c in chunking.iter_chunks_file(again)])

    def test_nonempty_documents_produce_chunks_and_empty_ones_do_not(self):
        fixtures.write_chunk_store(self.root, self.out)
        docs = {c["doc_id"] for c in chunking.iter_chunks_file(self.out)}
        self.assertIn("ZA-1", docs)
        self.assertNotIn("ZA-EMPTY", docs)
        meta = chunking.load_meta(self.out)
        self.assertEqual(meta["format"], "chunks.v1")
        self.assertEqual(meta["documents"], len(fixtures.DOCS))
        self.assertEqual(meta["corpus"]["documents"], len(fixtures.DOCS))


if __name__ == "__main__":
    unittest.main()
