"""Faithful character chunking with stable identifiers.

Chunks are character windows over ``Corpus.text(doc_id)``. Tokenisation for
retrieval happens later and never rewrites the stored text, so every
``chunk['text']`` is exactly ``text[start:end]``.
"""
from __future__ import annotations

import json
from pathlib import Path

from retrieval import contracts
from retrieval.corpus_adapter import CorpusAdapter

CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


def span_chunks(text: str, size: int, overlap: int, min_chars: int) -> list:
    """Return deterministic (start, end) windows over ``text``.

    Windows end on a whitespace boundary when a reasonable one exists, which
    keeps quotes readable without changing a single character of the source.
    """
    if size <= 0:
        raise contracts.ContractError("size must be positive")
    if not 0 <= overlap < size:
        raise contracts.ContractError("overlap must be in [0, size)")
    if min_chars <= 0 or min_chars > size:
        raise contracts.ContractError("min_chars must be in (0, size]")
    n = len(text)
    if n == 0:
        return []
    spans = []
    pos = 0
    while pos < n:
        end = min(pos + size, n)
        if end < n:
            floor = max(pos + min_chars, pos + (size * 3) // 5)
            cut = max(text.rfind("\n", floor, end), text.rfind(" ", floor, end))
            if cut <= floor:
                cut = -1
            if cut > 0:
                end = cut + 1
        spans.append((pos, end))
        if end >= n:
            break
        nxt = end - overlap
        pos = nxt if nxt > pos else pos + 1
    if not spans:
        spans.append((0, n))
    last_start, last_end = spans[-1]
    if last_end - last_start < min_chars and n > min_chars and len(spans) > 1:
        spans[-1] = (n - min_chars, n)
        if spans[-1] == spans[-2]:
            spans.pop()
    return spans


def iter_chunks(corpus: CorpusAdapter, chunking: dict, doc_ids=None, progress=None):
    """Yield chunk dictionaries for every non-empty document, in corpus order."""
    size = chunking["size"]
    overlap = chunking["overlap"]
    min_chars = chunking["min_chars"]
    ids = doc_ids if doc_ids is not None else corpus.doc_ids()
    for index, doc_id in enumerate(ids, 1):
        text = corpus.text(doc_id)
        for ordinal, (start, end) in enumerate(span_chunks(text, size, overlap, min_chars)):
            yield contracts.make_chunk(doc_id, ordinal, start, end, text[start:end])
        if progress is not None:
            progress(index, len(ids), doc_id)


def write_chunks(corpus_root=None, out_dir=None, settings=None, limit=None,
                 progress=None) -> dict:
    """Build ``chunks.jsonl`` plus ``meta.json``; returns build statistics."""
    settings = settings or {}
    chunking = contracts.chunking_config(settings)
    out_dir = Path(out_dir or settings.get("chunks_dir") or "artifacts/chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = CorpusAdapter(corpus_root)
    doc_ids = corpus.doc_ids()
    if limit is not None:
        doc_ids = doc_ids[:limit]
    count = 0
    total_chars = 0
    path = out_dir / CHUNKS_FILE
    temp = out_dir / (CHUNKS_FILE + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in iter_chunks(corpus, chunking, doc_ids, progress):
            problems = contracts.validate_chunk(chunk, corpus.text(chunk["doc_id"]))
            if problems:
                raise contracts.ContractError(f"{chunk['chunk_id']}: {problems}")
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1
            total_chars += len(chunk["text"])
    temp.replace(path)
    summary = corpus.summary()
    meta = {
        "format": contracts.CHUNK_FORMAT_VERSION,
        "chunking": chunking,
        "chunking_fingerprint": contracts.fingerprint(chunking),
        "documents": len(doc_ids),
        "chunks": count,
        "chunk_chars": total_chars,
        "corpus": summary,
        "source": "corpus text only; questions and labels are never read",
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    return {"out_dir": str(out_dir), "chunks": count, "documents": len(doc_ids),
            "chunk_chars": total_chars, "meta": meta}


def load_meta(chunks_dir=None) -> dict:
    path = Path(chunks_dir or "artifacts/chunks") / META_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def iter_chunks_file(path) -> "iter":
    path = Path(path)
    if path.is_dir():
        path = path / CHUNKS_FILE
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_faithful(chunks_iter, corpus: CorpusAdapter, sample=None) -> dict:
    """Verify ``chunk.text == Corpus.text(doc_id)[start:end]``; returns stats."""
    checked = 0
    bad = []
    for chunk in chunks_iter:
        if sample is not None and checked >= sample:
            break
        problems = contracts.validate_chunk(chunk, corpus.text(chunk["doc_id"]))
        checked += 1
        if problems:
            bad.append({"chunk_id": chunk.get("chunk_id"), "problems": problems})
    return {"checked": checked, "bad": bad[:20], "ok": not bad}
