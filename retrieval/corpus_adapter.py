"""Read-only access to the course corpus with a stable summary.

Wraps ``ir2026.Corpus`` so that every retrieval module uses the same text
accessor (``Corpus.text``), which defines the character offsets used by chunks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ir2026 import Corpus, DataError


class CorpusAdapter:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "corpus"
        self._corpus = Corpus(self.root)
        self._cache = {}

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    def doc_ids(self):
        return list(self._corpus.docs)

    def meta(self, doc_id: str) -> dict:
        if doc_id not in self._corpus.docs:
            raise DataError(f"Unknown document ID: {doc_id}")
        return self._corpus.docs[doc_id]

    def text(self, doc_id: str) -> str:
        if doc_id in self._cache:
            return self._cache[doc_id]
        text = self._corpus.text(doc_id)
        self._cache[doc_id] = text
        return text

    def length(self, doc_id: str) -> int:
        return len(self.text(doc_id))

    def summary(self) -> dict:
        """Corpus digest used to detect stale artefacts without re-reading text."""
        manifest = self.root / "documents.jsonl"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        return {"documents": len(self._corpus.docs),
                "manifest_sha256": digest,
                "root": str(self.root)}
