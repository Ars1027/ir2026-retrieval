"""Fictional fixtures for member B tests (no course data)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCS = {
    "VB-1": (
        "The Halden tidal mill was built in 1842 beside the Ostra inlet. "
        "Millers ground barley for the coastal villages until the flood of 1903.\n\n"
        "A wooden sluice gate replaced the original stone channel in 1871. "
        "The mill wheel is preserved in the Ostra maritime museum. "
    ),
    "VB-2": (
        "The Ostra maritime museum opened in 1968 in a converted salt warehouse. "
        "Its collection includes the Halden mill wheel and charts of the inlet. "
        "Opening hours were extended in 1994 after a renovation. "
    ),
    "VB-3": (
        "Barley was the main cereal of the coastal villages; rye arrived later. "
        "Millers were paid in grain, and the miller's house stood beside the sluice. "
    ),
    "VB-4": (
        "Flood records of the Ostra inlet mention 1903 and 1954. "
        "The 1903 flood destroyed two bridges and damaged the mill race. "
    ),
}


def write_corpus(root) -> Path:
    root = Path(root)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    rows = []
    for doc_id, text in DOCS.items():
        raw = text.encode("utf-8")
        (root / "docs" / f"{doc_id}.txt").write_bytes(raw)
        rows.append({"doc_id": doc_id, "path": f"docs/{doc_id}.txt",
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "chars": len(text), "bytes": len(raw), "url": "https://example.invalid/vb"})
    with (root / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return root


def write_chunk_store(corpus_root, out_dir, size=200, overlap=40, min_chars=20) -> dict:
    from retrieval import chunking
    settings = {"chunking": {"size": size, "overlap": overlap, "min_chars": min_chars}}
    return chunking.write_chunks(corpus_root=corpus_root, out_dir=out_dir, settings=settings)


def build_bm25_index(corpus_root, work_dir) -> Path:
    """Chunk store + BM25 index for integration tests; returns the index dir."""
    from retrieval import bm25
    work_dir = Path(work_dir)
    chunks_dir = work_dir / "chunks"
    write_chunk_store(corpus_root, chunks_dir)
    index_dir = work_dir / "bm25"
    bm25.build_index(chunks_path=chunks_dir, out_dir=index_dir,
                     params={"k1": 0.9, "b": 0.4, "use_stopwords": False})
    return index_dir


class FakeCorpus:
    def __init__(self, texts):
        self.texts = dict(texts)
        self.docs = {doc_id: {"doc_id": doc_id} for doc_id in self.texts}

    def text(self, doc_id):
        if doc_id not in self.texts:
            raise KeyError(doc_id)
        return self.texts[doc_id]


class StubVectorRetriever:
    """Deterministic stand-in for a plugged-in vector backend."""

    name = "stub-vector"

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.hits[:top_k]


def vector_hits(doc_ids):
    return [{"doc_id": doc_id, "rank": rank, "score": 1.0 / rank}
            for rank, doc_id in enumerate(doc_ids, 1)]
