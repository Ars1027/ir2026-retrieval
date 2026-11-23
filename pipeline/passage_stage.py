"""Opt-in passage reranking for the shared retrieval pipeline."""
from __future__ import annotations

import time


def candidate_depth(settings: dict, final_docs: int) -> int:
    # 关闭重排时沿用原来的文档返回数量。
    options = settings.get("passage_rerank") or {}
    if options.get("mode", "off") == "off":
        return final_docs
    return options["candidate_docs"]


def rerank(query: str, docs: list, tools, settings: dict, final_docs: int) -> list:
    # 只有启用时才加载重排实现，普通 BM25 和混合配置沿用原路径。
    config = settings.get("passage_rerank") or {}
    if config.get("mode", "off") == "off":
        return docs[:final_docs]

    from retrieval.passage_rank import rerank_documents

    # 段落评分通过原有语料接口读取本地正文，沿用配置中的停用词开关。
    started = time.monotonic()
    use_stopwords = settings["bm25"]["use_stopwords"]
    options = dict(config, use_stopwords=use_stopwords)
    ranked, details = rerank_documents(query, docs, tools.corpus.text, options)
    elapsed_ms = (time.monotonic() - started) * 1000

    # 保存候选池和各级名次，供区分召回遗漏、排序遗漏和消融影响。
    tools.record_retrieval(
        stage="passage_rerank",
        arguments={
            "mode": options["mode"],
            "candidate_docs": options["candidate_docs"],
            "words": options["words"],
            "stride": options["stride"],
            "rrf_k": options["rrf_k"],
        },
        output={
            "documents": details,
            "returned": min(len(ranked), final_docs),
        },
        doc_ids=[doc["doc_id"] for doc in ranked],
        elapsed_ms=elapsed_ms,
    )
    return ranked[:final_docs]
