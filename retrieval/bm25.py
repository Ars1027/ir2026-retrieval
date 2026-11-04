"""BM25 inverted index over faithful chunks.

Design notes
------------
* Postings are stored as parallel NumPy arrays (chunk id, term frequency) plus
  an offset table, so a query only touches the relevant slices.
* Tokenisation is kept in one place and is never written back to the corpus:
  case folding, comma removal inside numbers and stopword filtering only affect
  the index copy of the text.
* The index is built once by ``scripts/build_bm25.py`` and reloaded per query;
  queries never rebuild it.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np

from retrieval import contracts
from retrieval.chunking import META_FILE as CHUNK_META_FILE

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._'-][a-z0-9]+)*")
TOKEN_RE_CI = re.compile(TOKEN_RE.pattern, re.IGNORECASE)
NUMBER_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

STOPWORDS = frozenset("""a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down during each few for from
further had has have having he her here hers herself him himself his how i if in into is it its itself just
me more most my myself no nor not of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves s t""".split())

INDEX_VERSION = "bm25.v1"
DEFAULT_PARAMS = {"k1": 0.9, "b": 0.4, "use_stopwords": True, "capitalized_boost": 1.25}
META_FILE = "meta.json"
VOCAB_FILE = "vocab.json"
ARRAY_FILES = ("postings_chunk.npy", "postings_tf.npy", "offsets.npy",
               "doc_len.npy", "chunk_starts.npy", "chunk_ends.npy")
CHUNK_IDS_FILE = "chunk_ids.json"


def normalize_numbers(text: str) -> str:
    return NUMBER_COMMA_RE.sub("", text)


def tokenize(text: str, use_stopwords: bool = True) -> list:
    """Lowercased word tokens; entities and years survive as ordinary tokens."""
    tokens = TOKEN_RE.findall(normalize_numbers(text).lower())
    if use_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens


def tokenize_query(query: str, params: dict) -> list:
    """Return (term, weight) pairs, boosting capitalised query tokens."""
    use_stop = bool(params.get("use_stopwords", True))
    boost = float(params.get("capitalized_boost", 1.0) or 1.0)
    out = {}
    for match in TOKEN_RE_CI.finditer(normalize_numbers(query)):
        raw = match.group(0)
        term = raw.lower()
        if use_stop and term in STOPWORDS:
            continue
        weight = boost if (boost != 1.0 and raw[:1].isupper() and len(raw) > 2) else 1.0
        out[term] = max(out.get(term, 0.0), weight)
    return list(out.items())


def idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


class BM25Index:
    """Persistent BM25 index; load once per process, then answer many queries."""

    def __init__(self, index_dir=None):
        self.index_dir = Path(index_dir or "artifacts/bm25")
        self.ready = False
        self.meta = {}
        self.vocab = {}
        self.chunk_ids = []
        self._arrays = {}

    # -- persistence -----------------------------------------------------
    def load(self):
        if not self.index_dir.is_dir():
            raise FileNotFoundError(f"BM25 index directory missing: {self.index_dir}; "
                                    "run scripts/build_bm25.py")
        self.meta = json.loads((self.index_dir / META_FILE).read_text(encoding="utf-8"))
        self.vocab = json.loads((self.index_dir / VOCAB_FILE).read_text(encoding="utf-8"))
        self.chunk_ids = json.loads((self.index_dir / CHUNK_IDS_FILE).read_text(encoding="utf-8"))
        for name in ARRAY_FILES:
            key = name[:-4]
            self._arrays[key] = np.load(self.index_dir / name, mmap_mode="r")
        params = {**DEFAULT_PARAMS, **(self.meta.get("params") or {})}
        self.params = params
        self.k1 = float(params["k1"])
        self.b = float(params["b"])
        self.n_chunks = len(self.chunk_ids)
        self.avgdl = float(self.meta["avg_chunk_len"]) or 1.0
        if self.n_chunks == 0:
            raise ValueError("empty BM25 index")
        self.ready = True
        return self

    def save(self, out_dir=None, params=None, build_stats=None) -> dict:
        out_dir = Path(out_dir or self.index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / VOCAB_FILE).write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")
        (out_dir / CHUNK_IDS_FILE).write_text(json.dumps(self.chunk_ids, ensure_ascii=False),
                                              encoding="utf-8")
        for name in ARRAY_FILES:
            np.save(out_dir / name, self._arrays[name[:-4]])
        meta = dict(self.meta)
        meta.update({"format": INDEX_VERSION, "params": params or self.params,
                     "chunks": self.n_chunks, "terms": len(self.vocab),
                     "avg_chunk_len": self.avgdl, "build_stats": build_stats or {}})
        (out_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
        self.meta = meta
        self.index_dir = out_dir
        return meta

    # -- search ----------------------------------------------------------
    def search_hits(self, query: str, top_k: int = 200) -> list:
        """Return raw BM25 hits with ranks starting at 1."""
        if not self.ready:
            raise RuntimeError("BM25Index.load() must be called before search")
        if not isinstance(query, str) or not query.strip():
            return []
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        terms = tokenize_query(query, self.params)
        if not terms:
            return []
        offsets = self._arrays["offsets"]
        postings = self._arrays["postings_chunk"]
        tfs = self._arrays["postings_tf"]
        doc_len = self._arrays["doc_len"]
        scores = np.zeros(self.n_chunks, dtype=np.float32)
        n = self.n_chunks
        matched = False
        for term, weight in terms:
            tid = self.vocab.get(term)
            if tid is None:
                continue
            lo, hi = int(offsets[tid]), int(offsets[tid + 1])
            if hi <= lo:
                continue
            ids = np.asarray(postings[lo:hi], dtype=np.int32)
            tf = np.asarray(tfs[lo:hi], dtype=np.float32)
            df = hi - lo
            denom = tf + self.k1 * (1.0 - self.b + self.b * np.asarray(doc_len[ids], dtype=np.float32) / self.avgdl)
            scores[ids] += (weight * idf(n, df)) * (tf * (self.k1 + 1.0) / denom)
            matched = True
        if not matched:
            return []
        positive = np.flatnonzero(scores > 0)
        if positive.size == 0:
            return []
        k = min(int(top_k), positive.size)
        if k < positive.size:
            part = np.argpartition(scores[positive], -k)[-k:]
            candidates = positive[part]
        else:
            candidates = positive
        # Deterministic ordering: score desc, then chunk_id asc.
        ordered = sorted(candidates.tolist(), key=lambda i: (-float(scores[i]), self.chunk_ids[i]))[:k]
        starts = self._arrays["chunk_starts"]
        ends = self._arrays["chunk_ends"]
        hits = []
        for rank, idx in enumerate(ordered, 1):
            cid = self.chunk_ids[idx]
            doc_id, _, _ = cid.rpartition(contracts.CHUNK_ID_SEPARATOR)
            hits.append({"chunk_id": cid, "doc_id": doc_id,
                         "start": int(starts[idx]), "end": int(ends[idx]),
                         "score": float(scores[idx]), "rank": rank, "source": "bm25"})
        return hits


class BM25Retriever:
    """Contract-facing retriever: ``search(query, top_k) -> list[Hit]``."""

    source = "bm25"

    def __init__(self, index_dir=None, settings=None):
        self.settings = settings or {}
        self.index_dir = index_dir or self.settings.get("bm25_index_dir")
        self.index = BM25Index(self.index_dir).load()
        self.load_seconds = None

    def search(self, query: str, top_k: int) -> list:
        return self.index.search_hits(query, top_k)

    def describe(self) -> dict:
        return {"source": self.source, "index_dir": str(self.index_dir),
                "chunks": self.index.n_chunks, "terms": len(self.index.vocab),
                "params": self.index.params}


def build_index(chunks_path=None, out_dir=None, params=None, limit=None,
                progress=None) -> dict:
    """Build the index from ``chunks.jsonl``; labels and questions are never read."""
    from retrieval.chunking import iter_chunks_file

    params = {**DEFAULT_PARAMS, **(params or {})}
    chunks_path = Path(chunks_path or "artifacts/chunks")
    out_dir = Path(out_dir or "artifacts/bm25")
    use_stop = bool(params.get("use_stopwords", True))
    started = time.monotonic()

    # Pass 1: vocabulary, document frequency and chunk metadata.
    vocab = {}
    df_counts = []
    chunk_ids_list = []
    starts = []
    ends = []
    for count, chunk in enumerate(iter_chunks_file(chunks_path), 1):
        if limit is not None and count > limit:
            break
        counts = {}
        for token in tokenize(chunk["text"], use_stop):
            tid = vocab.get(token)
            if tid is None:
                tid = len(vocab)
                vocab[token] = tid
                df_counts.append(0)
            counts[tid] = counts.get(tid, 0) + 1
        for tid in counts:
            df_counts[tid] += 1
        chunk_ids_list.append(chunk["chunk_id"])
        starts.append(chunk["start"])
        ends.append(chunk["end"])
        if progress is not None and count % 20000 == 0:
            progress(count)

    n = len(chunk_ids_list)
    if n == 0:
        raise ValueError(f"no chunks found in {chunks_path}")

    # Pass 2: postings, keeping memory bounded by the postings themselves.
    term_ids = []
    postings_chunk = []
    tfs = []
    lengths = np.zeros(n, dtype=np.int32)
    for idx, chunk in enumerate(iter_chunks_file(chunks_path)):
        if idx >= n:
            break
        counts = {}
        for token in tokenize(chunk["text"], use_stop):
            tid = vocab[token]
            counts[tid] = counts.get(tid, 0) + 1
        if not counts:
            continue
        keys = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
        vals = np.fromiter(counts.values(), dtype=np.int32, count=len(counts))
        term_ids.append(keys)
        tfs.append(np.minimum(vals, np.iinfo(np.uint16).max).astype(np.uint16))
        postings_chunk.append(np.full(keys.shape, idx, dtype=np.int32))
        lengths[idx] = int(vals.sum())
    if not term_ids:
        raise ValueError("all chunks are empty after tokenisation")
    term_arr = np.concatenate(term_ids)
    chunk_arr = np.concatenate(postings_chunk)
    tf_arr = np.concatenate(tfs)
    order = np.argsort(term_arr, kind="stable")
    term_arr = term_arr[order]
    chunk_arr = chunk_arr[order]
    tf_arr = tf_arr[order]
    offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
    counts_per_term = np.bincount(term_arr, minlength=len(vocab))
    offsets[1:] = np.cumsum(counts_per_term)

    index = BM25Index(out_dir)
    index.vocab = vocab
    index.chunk_ids = chunk_ids_list
    index.n_chunks = n
    index.avgdl = float(lengths.mean()) if lengths.mean() > 0 else 1.0
    index.k1 = float(params["k1"])
    index.b = float(params["b"])
    index.params = params
    index.meta = {
        "corpus": _read_chunk_meta(chunks_path),
        "chunking_fingerprint": None,
        "postings": int(term_arr.size),
        "source": "chunks.jsonl only",
    }
    chunk_meta_path = Path(chunks_path)
    chunk_meta_path = chunk_meta_path / CHUNK_META_FILE if chunk_meta_path.is_dir() else chunk_meta_path.parent / CHUNK_META_FILE
    if chunk_meta_path.is_file():
        chunk_meta = json.loads(chunk_meta_path.read_text(encoding="utf-8"))
        index.meta["chunking_fingerprint"] = chunk_meta.get("chunking_fingerprint")
        index.meta["corpus"] = chunk_meta.get("corpus")
    index._arrays = {"postings_chunk": chunk_arr.astype(np.int32),
                     "postings_tf": tf_arr,
                     "offsets": offsets,
                     "doc_len": lengths,
                     "chunk_starts": np.array(starts, dtype=np.int32),
                     "chunk_ends": np.array(ends, dtype=np.int32)}
    build_stats = {"chunks": n, "terms": len(vocab), "postings": int(term_arr.size),
                   "build_seconds": round(time.monotonic() - started, 3)}
    index.save(out_dir=out_dir, params=params, build_stats=build_stats)
    build_stats["index_bytes"] = sum(p.stat().st_size for p in out_dir.glob("*") if p.is_file())
    return build_stats


def _read_chunk_meta(chunks_path) -> dict:
    path = Path(chunks_path)
    path = path / CHUNK_META_FILE if path.is_dir() else path.parent / CHUNK_META_FILE
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("corpus", {})
        except ValueError:
            return {}
    return {}


def index_size(index_dir=None) -> int:
    path = Path(index_dir or "artifacts/bm25")
    return sum(p.stat().st_size for p in path.glob("*") if p.is_file())
