"""Model-directed indexed search, with the course's real model_tool logging."""
from __future__ import annotations

from ir2026 import CorpusTools, DataError
from retrieval.document_rank import aggregate_documents


class IndexedTools(CorpusTools):
    def __init__(self, supplied, retriever, settings):
        if supplied.method_type != "agent":
            raise DataError("posttraining policy requires method_type=agent")
        super().__init__(supplied.corpus, supplied.qid, supplied.limits,
                         supplied.trace_path, method_type="agent")
        # The course worker owns this list and persists it on failure/timeout.
        self.trace = supplied.trace
        self.started = supplied.started
        self.retriever = retriever
        self.settings = settings
        self.observed = {}
        self.latest_ranking = []

    def _remember(self, doc_id, start, text):
        if len(text) >= self.limits["min_quote_chars"]:
            self.observed[doc_id] = {"doc_id": doc_id, "start": start,
                                     "end": start + len(text), "text": text}

    def search(self, query):
        started = self._begin()
        try:
            hits = self.retriever.search(query, self.settings.get("candidate_k", 200))
            docs = aggregate_documents(hits, top_k=8)
            passages = []
            for doc in docs:
                hit = doc["hits"][0]
                start = hit["start"]
                text = self.corpus.text(doc["doc_id"])[start:min(hit["end"], start + 650)]
                if len(text) < self.limits["min_quote_chars"]:
                    continue
                row = {"doc_id": doc["doc_id"], "rank": doc["rank"],
                       "start": start, "end": start + len(text), "text": text}
                passages.append(row)
            output = {"passages": passages}
            # Only passages actually returned to the model count as observed.
            self._record("search", {"query": query}, output,
                         [r["doc_id"] for r in passages], started)
            for row in passages:
                self._remember(row["doc_id"], row["start"], row["text"])
            self.latest_ranking = list(dict.fromkeys([r["doc_id"] for r in passages] + self.latest_ranking))
            return output
        except (ValueError, OSError) as exc:
            self._record("search", {"query": query}, {"error": str(exc)}, [], started, str(exc))
            raise

    def read_observed(self, doc_id, start, end=None):
        if doc_id not in self.observed:
            raise ValueError("read may only open a document returned by search")
        max_chars = min(1000, end - start) if end is not None else 1000
        result = self.read(doc_id, start=start, max_chars=max_chars)
        self._remember(doc_id, result["start"], result["text"])
        return result

    def finish(self, action):
        ids = action["doc_ids"]
        if any(doc_id not in self.observed for doc_id in ids):
            raise ValueError("finish may only return observed document IDs with evidence")
        quotes = [{"doc_id": doc_id, "text": self.observed[doc_id]["text"]}
                  for doc_id in ids[:self.limits["max_evidence_quotes"]]]
        if action["answer"].strip() and not quotes:
            raise ValueError("an answer needs observed evidence")
        return {"answer": action["answer"].strip(), "doc_ids": ids, "evidence": quotes,
                "status": "ok" if action["answer"].strip() else "no_answer"}
