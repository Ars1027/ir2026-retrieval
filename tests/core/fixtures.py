"""Fictional synthetic fixtures for core tests.

The content is invented for testing and contains no course data.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCS = {
    "ZA-1": (
        "The Zarnia Observatory was founded in 1907 by Ilse Marrow near the town of Belport. "
        "Its first director catalogued the Pelican Nebula using a brass refractor.\n\n"
        "In 1954 the observatory was moved to the Kestrel Ridge plateau after a storm damaged the dome. "
        "The move was funded by the Marrow family trust and took eleven months. "
    ),
    "ZA-2": (
        "Belport became a trading port in 1712. "
        "Shipwrights from the northern isles built the first dry dock, and the harbour handled wool and tin. "
        "A customs house was added in 1788; its ledger is kept in the Kestrel Ridge archive. "
    ),
    "ZA-3": (
        "The Kestrel Ridge plateau is a narrow limestone shelf above the Belport estuary. "
        "Weather stations there record strong westerly winds between October and March. "
        "The plateau was surveyed in 1899 by the cartographer Oda Vell. "
    ),
    "ZA-4": (
        "Short note about the archive.\n"
        "The Kestrel Ridge archive holds the 1788 customs ledger, the 1907 observatory charter and "
        "photographs of the brass refractor. Access requires a written request to the trust. "
    ),
    "ZA-EMPTY": "",
}


def write_corpus(root) -> Path:
    """Create a miniature corpus with the same layout as the course data."""
    root = Path(root)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    rows = []
    for doc_id, text in DOCS.items():
        raw = text.encode("utf-8")
        (root / "docs" / f"{doc_id}.txt").write_bytes(raw)
        rows.append({"doc_id": doc_id, "path": f"docs/{doc_id}.txt",
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "chars": len(text), "bytes": len(raw), "url": "https://example.invalid/doc"})
    with (root / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return root


def chunk_settings(size=120, overlap=20, min_chars=20) -> dict:
    return {"chunking": {"size": size, "overlap": overlap, "min_chars": min_chars}}


def write_chunk_store(corpus_root, out_dir, **kwargs) -> dict:
    from retrieval import chunking
    settings = {"chunking": {"size": kwargs.pop("size", 120),
                             "overlap": kwargs.pop("overlap", 20),
                             "min_chars": kwargs.pop("min_chars", 20)}}
    return chunking.write_chunks(corpus_root=corpus_root, out_dir=out_dir, settings=settings)


def build_test_index(corpus_root, workdir, **params) -> tuple:
    """Build chunks + BM25 index; returns (chunks_dir, index_dir, stats)."""
    workdir = Path(workdir)
    chunks_dir = workdir / "chunks"
    index_dir = workdir / "bm25"
    write_chunk_store(corpus_root, chunks_dir)
    from retrieval import bm25
    stats = bm25.build_index(chunks_dir, index_dir, params)
    return chunks_dir, index_dir, stats


class FakeCorpus:
    """Minimal corpus object exposing ``text(doc_id)`` for assembler tests."""

    def __init__(self, texts):
        self.texts = dict(texts)
        self.docs = {doc_id: {"doc_id": doc_id} for doc_id in self.texts}

    def text(self, doc_id):
        if doc_id not in self.texts:
            raise KeyError(doc_id)
        return self.texts[doc_id]
