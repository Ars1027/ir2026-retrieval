"""Candidate-local passage BM25 and optional reciprocal rank fusion."""
from __future__ import annotations

from contextlib import closing
import re
import sqlite3

from retrieval.bm25 import normalize_numbers, tokenize

_WORD = re.compile(r"\S+")


def _windows(text: str, words: int, stride: int):
    tokens = list(_WORD.finditer(text))
    for first in range(0, len(tokens), stride):
        stop = min(first + words, len(tokens))
        start = tokens[first].start()
        end = tokens[stop - 1].end()
        yield start, end
        if stop == len(tokens):
            break


def _best_passages(docs: list, text_getter, expression: str, words: int, stride: int) -> dict:
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("""
            CREATE VIRTUAL TABLE passages USING fts5(
                text, doc_id UNINDEXED, start UNINDEXED, end UNINDEXED,
                tokenize='porter unicode61'
            )
        """)
        insert = "INSERT INTO passages VALUES (?, ?, ?, ?)"
        for doc in docs:
            doc_id = doc["doc_id"]
            text = text_getter(doc_id)
            rows = []
            for start, end in _windows(text, words, stride):
                normalized = normalize_numbers(text[start:end])
                rows.append((normalized, doc_id, start, end))
            db.executemany(insert, rows)

        # SQLite BM25 返回负分。先保留每篇最高分，再按原文位置消歧。
        rows = db.execute("""
            SELECT doc_id, start, end, bm25(passages)
            FROM passages
            WHERE passages MATCH ?
        """, (expression,))
        best = {}
        for doc_id, start, end, raw_score in rows:
            score = (-raw_score, -start, end)
            previous = best.get(doc_id)
            if previous is None or score > previous:
                best[doc_id] = score
    return best


def rerank_documents(query: str, docs: list, text_getter, options: dict) -> tuple[list, list]:
    """Rerank the shared candidate set, keeping the original evidence hits."""
    terms = tokenize(query, use_stopwords=options["use_stopwords"])
    terms = list(dict.fromkeys(terms))
    if not docs or not terms:
        return docs, []

    expression = " OR ".join(f'"{term}"' for term in terms)
    best = _best_passages(
        docs, text_getter, expression, options["words"], options["stride"],
    )

    # 无匹配段落的候选按零分计，段落同分时保留原候选顺序。
    def passage_key(doc):
        match = best.get(doc["doc_id"], (0.0, 0, 0))
        return -match[0], doc["rank"]

    passage_order = sorted(docs, key=passage_key)
    passage_ranks = {
        doc["doc_id"]: rank
        for rank, doc in enumerate(passage_order, 1)
    }

    def fusion_score(doc):
        base_rank = doc["rank"]
        passage_rank = passage_ranks[doc["doc_id"]]
        constant = options["rrf_k"]
        return 1 / (constant + base_rank) + 1 / (constant + passage_rank)

    def fusion_key(doc):
        score = fusion_score(doc)
        return -score, doc["rank"]

    use_fusion = options["mode"] == "fusion"
    final_order = sorted(docs, key=fusion_key) if use_fusion else passage_order

    # 新排名保留原始 hit 列表，让三组对同一文档使用相同的证据定位。
    ranked = []
    details = []
    for rank, doc in enumerate(final_order, 1):
        doc_id = doc["doc_id"]
        match = best.get(doc_id)
        passage_score = match[0] if match is not None else 0.0
        score = fusion_score(doc) if use_fusion else passage_score
        ranked.append(dict(doc, rank=rank, score=score))

        span = [-match[1], match[2]] if match is not None else None
        details.append({
            "doc_id": doc_id,
            "base_rank": doc["rank"],
            "passage_rank": passage_ranks[doc_id],
            "passage_score": passage_score,
            "passage_span": span,
            "rank": rank,
            "score": score,
        })
    return ranked, details
