"""Document-level reciprocal rank fusion; raw retrieval scores stay in hits."""
def fuse_rrf(bm25_docs, dense_docs, k, top_k):
    if k < 0:
        raise ValueError("rrf_k must be nonnegative")
    merged = {}
    for source, docs in (("bm25", bm25_docs), ("dense", dense_docs)):
        seen = set()
        for doc in docs:
            doc_id = doc["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank = doc["rank"]
            if rank < 1:
                raise ValueError("document ranks start at 1")
            item = merged.setdefault(doc_id, {"doc_id": doc_id, "score": 0.0, "hits": [], "source_ranks": {}})
            item["score"] += 1.0 / (k + rank)
            item["source_ranks"][source] = rank
            item["hits"].extend(dict(hit) for hit in doc["hits"])
    ordered = sorted(merged.values(), key=lambda d: (-d["score"], d["doc_id"]))
    return [dict(doc, rank=i) for i, doc in enumerate(ordered[:max(0, min(10, top_k))], 1)]
