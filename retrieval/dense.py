"""Dense chunk retrieval over FAISS exact inner product (member B).

Vectors are built offline from member A's chunk store. The index stores the
encoder description, the chunking fingerprint and the corpus digest so a stale
or mismatched index is detected instead of being used silently.

FAISS ``IndexFlatIP`` is used when the package is importable; otherwise the same
scoring is computed with NumPy and recorded in the index metadata.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from retrieval import contracts
from retrieval.chunking import META_FILE as CHUNK_META_FILE
from retrieval.chunking import iter_chunks_file
from retrieval.embedding import Encoder, EncoderError

INDEX_VERSION = "dense.v1"
VECTORS_FILE = "vectors.npy"
FAISS_FILE = "index.faiss"
META_FILE = "meta.json"
CHUNK_IDS_FILE = "chunk_ids.json"
SPANS_FILE = "spans.npy"

try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - depends on the installation
    faiss = None


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            count += block.count(b"\n")
    return count


def _chunk_meta(chunks_path: Path) -> dict:
    path = chunks_path / CHUNK_META_FILE if chunks_path.is_dir() else chunks_path.parent / CHUNK_META_FILE
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def build_index(chunks_path=None, out_dir=None, settings=None, limit=None,
                encoder=None, progress=None) -> dict:
    """Encode every chunk and persist the vectors plus metadata."""
    settings = settings or {}
    chunks_path = Path(chunks_path or settings.get("chunks_dir") or "artifacts/member_a/chunks")
    out_dir = Path(out_dir or settings.get("dense_index_dir") or "artifacts/member_b/dense")
    chunks_file = chunks_path / "chunks.jsonl" if chunks_path.is_dir() else chunks_path
    if not chunks_file.is_file():
        raise FileNotFoundError(f"chunk store missing: {chunks_file}; run scripts/build_chunks.py")
    chunk_meta = _chunk_meta(chunks_path)

    encoder = encoder or Encoder(settings=settings)
    started = time.monotonic()
    total = _count_lines(chunks_file)
    if limit is not None:
        total = min(total, limit)
    if total == 0:
        raise ValueError(f"no chunks to encode in {chunks_file}")
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors = np.lib.format.open_memmap(out_dir / VECTORS_FILE, mode="w+",
                                       dtype=np.float32, shape=(total, encoder.dimension))
    chunk_ids = []
    spans = np.zeros((total, 2), dtype=np.int64)
    filled = 0
    encode_seconds = 0.0
    texts = []
    meta_iter = (chunk for chunk in iter_chunks_file(chunks_file))
    while filled < total:
        chunk = next(meta_iter, None)
        if chunk is None:
            break
        texts.append((chunk["chunk_id"], chunk["start"], chunk["end"], chunk["text"]))
        if len(texts) >= min(256, total - filled) or len(texts) >= 256:
            batch_started = time.monotonic()
            matrix = encoder.encode_documents([item[3] for item in texts])
            encode_seconds += time.monotonic() - batch_started
            for offset, item in enumerate(texts):
                vectors[filled + offset] = matrix[offset]
                chunk_ids.append(item[0])
                spans[filled + offset] = (item[1], item[2])
            filled += len(texts)
            texts = []
            if progress is not None and filled % 20000 < 256:
                progress(filled, total)
    if filled != total:
        raise RuntimeError(f"encoded {filled} of {total} chunks; refusing to write a partial index")
    vectors.flush()
    del vectors
    np.save(out_dir / SPANS_FILE, spans)
    (out_dir / CHUNK_IDS_FILE).write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")

    index_type = "numpy-exact-ip"
    if faiss is not None:
        matrix = np.load(out_dir / VECTORS_FILE, mmap_mode="r")
        flat = faiss.IndexFlatIP(encoder.dimension)
        batch = 50000
        for start in range(0, filled, batch):
            flat.add(np.ascontiguousarray(matrix[start:start + batch]))
        faiss.write_index(flat, str(out_dir / FAISS_FILE))
        index_type = "faiss-flat-ip"
        del matrix

    elapsed = time.monotonic() - started
    meta = {
        "format": INDEX_VERSION,
        "vectors": filled,
        "dimension": int(encoder.dimension),
        "encoder": encoder.describe(),
        "index_type": index_type,
        "normalize": bool(encoder.config.get("normalize", True)),
        "chunking_fingerprint": chunk_meta.get("chunking_fingerprint"),
        "corpus": chunk_meta.get("corpus"),
        "chunk_store": str(chunks_file),
        "build_stats": {"chunks": filled, "elapsed_seconds": round(elapsed, 3),
                        "encode_seconds": round(encode_seconds, 3),
                        "chunks_per_second": round(filled / max(encode_seconds, 1e-9), 2)},
        "source": "chunk store only; questions and labels are never read",
    }
    meta["index_bytes"] = None
    (out_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    meta["index_bytes"] = index_size(out_dir)
    (out_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    return meta


def index_size(index_dir=None) -> int:
    path = Path(index_dir or "artifacts/member_b/dense")
    return sum(p.stat().st_size for p in path.glob("*") if p.is_file())


class DenseIndex:
    def __init__(self, index_dir=None):
        self.index_dir = Path(index_dir or "artifacts/member_b/dense")
        self.meta = {}
        self.chunk_ids = []
        self.spans = None
        self.vectors = None
        self._faiss = None
        self.encoder = None

    def load(self, with_encoder=True):
        if not self.index_dir.is_dir():
            raise FileNotFoundError(f"dense index directory missing: {self.index_dir}; "
                                    "run scripts/build_dense.py")
        self.meta = json.loads((self.index_dir / META_FILE).read_text(encoding="utf-8"))
        self.chunk_ids = json.loads((self.index_dir / CHUNK_IDS_FILE).read_text(encoding="utf-8"))
        self.spans = np.load(self.index_dir / SPANS_FILE, mmap_mode="r")
        if self.meta.get("index_type") == "faiss-flat-ip" and faiss is not None:
            self._faiss = faiss.read_index(str(self.index_dir / FAISS_FILE))
        else:
            self.vectors = np.load(self.index_dir / VECTORS_FILE, mmap_mode="r")
        if with_encoder:
            footprint = dict(self.meta.get("encoder") or {})
            footprint.pop("fallback_reason", None)
            self.encoder = Encoder(footprint)
            if self.encoder.dimension != self.meta["dimension"]:
                raise EncoderError(
                    f"encoder dimension {self.encoder.dimension} does not match index "
                    f"dimension {self.meta['dimension']}; rebuild the dense index")
        return self

    @property
    def n_vectors(self) -> int:
        return len(self.chunk_ids)

    def search_vectors(self, query_vector: np.ndarray, top_k: int):
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        if self._faiss is not None:
            scores, rows = self._faiss.search(query, min(top_k, self.n_vectors))
            return scores[0], rows[0]
        scores = self.vectors @ query[0]
        rows = np.argsort(-scores, kind="stable")[:min(top_k, self.n_vectors)]
        return scores[rows], rows

    def search_hits(self, query: str, top_k: int = 200) -> list:
        if self.encoder is None:
            self.load()
        if not isinstance(query, str) or not query.strip():
            return []
        query_vector = self.encoder.encode_query(query)
        scores, rows = self.search_vectors(query_vector, top_k)
        hits = []
        for rank, (score, row) in enumerate(zip(scores, rows), 1):
            row = int(row)
            if row < 0:
                continue
            cid = self.chunk_ids[row]
            doc_id, _, _ = cid.rpartition(contracts.CHUNK_ID_SEPARATOR)
            hits.append({"chunk_id": cid, "doc_id": doc_id,
                         "start": int(self.spans[row][0]), "end": int(self.spans[row][1]),
                         "score": float(score), "rank": rank, "source": "dense"})
        return hits

    def describe(self) -> dict:
        return {"vectors": self.n_vectors,
                "dimension": self.meta.get("dimension"),
                "index_type": self.meta.get("index_type"),
                "encoder": self.meta.get("encoder"),
                "chunking_fingerprint": self.meta.get("chunking_fingerprint")}


class DenseRetriever:
    """Contract-facing retriever: ``search(query, top_k) -> list[Hit]``."""

    source = "dense"

    def __init__(self, index_dir=None, settings=None):
        self.settings = settings or {}
        self.index_dir = index_dir or self.settings.get("dense_index_dir")
        self.index = DenseIndex(self.index_dir).load()
        self.load_seconds = None

    def search(self, query: str, top_k: int) -> list:
        return self.index.search_hits(query, top_k)

    def describe(self) -> dict:
        return {"source": self.source, "index_dir": str(self.index_dir),
                **self.index.describe()}


def check_chunk_consistency(index_dir, chunks_dir) -> dict:
    """Compare the chunk fingerprint stored in an index with the current store."""
    meta = json.loads((Path(index_dir) / META_FILE).read_text(encoding="utf-8"))
    current = _chunk_meta(Path(chunks_dir)).get("chunking_fingerprint")
    stored = meta.get("chunking_fingerprint")
    return {"stored": stored, "current": current, "match": stored == current}
