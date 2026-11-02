"""Fixed retrieval pipeline consuming member A's public interfaces."""
import time
from retrieval.fusion import fuse_rrf


def retrieve(query, tools, settings, method="hybrid", embedding_usage=None):
    from retrieval.document_rank import aggregate_documents
    from retrieval.dense import DenseRetriever
    branches = {}
    for source in (("bm25", "dense") if method == "hybrid" else ("dense",)):
        started = time.monotonic()
        if source == "bm25":
            from retrieval.bm25 import BM25Retriever
            retriever = BM25Retriever(settings["bm25_index_dir"], settings)
        else:
            retriever = DenseRetriever(settings["dense_index_dir"], settings)
        hits = retriever.search(query, settings["candidate_k"])
        if source == "dense" and embedding_usage is not None:
            embedding_usage["input_tokens"] = retriever.last_query_tokens
        tools.record_retrieval(source + "_search", {"query": query, "top_k": settings["candidate_k"]}, {"hits": hits}, [h["doc_id"] for h in hits], (time.monotonic()-started)*1000)
        started = time.monotonic()
        branches[source] = aggregate_documents(hits, settings["candidate_k"])
        tools.record_retrieval(source + "_aggregate", {}, {"documents": branches[source]}, [d["doc_id"] for d in branches[source]], (time.monotonic()-started)*1000)
    if method == "dense":
        return branches["dense"][:10]
    started = time.monotonic()
    ranked = fuse_rrf(branches["bm25"], branches["dense"], settings["rrf_k"], 10)
    tools.record_retrieval("rrf", {"k": settings["rrf_k"]}, {"documents": ranked}, [d["doc_id"] for d in ranked], (time.monotonic()-started)*1000)
    return ranked


def solve_method(question, tools, config, method):
    from pipeline.settings import load_settings
    from pipeline.result_builder import assemble_result
    settings = load_settings()
    embedding_usage = {}
    ranked = retrieve(question["question"], tools, settings, method, embedding_usage)
    result = assemble_result(question, ranked, tools, settings)
    usage = result.setdefault("usage", {})
    answer_tokens = usage.get("model_input_tokens")
    query_tokens = embedding_usage.get("input_tokens")
    usage["model_input_tokens"] = answer_tokens + query_tokens if answer_tokens is not None and query_tokens is not None else None
    usage.setdefault("model_output_tokens", 0)
    usage["token_notes"] = (usage.get("token_notes", "") + " Embedding query tokens include instruction and tokenizer special tokens; unknown component totals remain null.").strip()
    return result
