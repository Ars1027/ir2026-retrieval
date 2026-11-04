"""API embeddings and exact chunk retrieval with document-level RRF."""
import base64
import json
import os
import time
import urllib.request
from pathlib import Path
import numpy as np
import faiss
from retrieval.interfaces import VectorRetriever, DocumentFusion
from pipeline.settings import resolve_path


class Encoder:
    def __init__(self, config):
        self.config = config
        self.last_tokens = None

    def encode(self, texts):
        key = os.environ[self.config.get("api_key_env", "POLOAI_API_KEY")]
        payload = {"model": self.config["model"], "input": texts, "encoding_format": "base64"}
        if self.config["base_url"] == "https://openrouter.ai/api/v1":
            payload["provider"] = {"order": ["deepinfra", "parasail"]}
        body = json.dumps(payload).encode()
        request = urllib.request.Request(self.config["base_url"].rstrip("/")+"/embeddings", data=body,
            headers={"Content-Type":"application/json", "Authorization":"Bearer "+key})
        with urllib.request.urlopen(request, timeout=self.config.get("timeout_seconds",120)) as response:
            result=json.load(response)
        rows=sorted(result["data"], key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(texts))):
            raise ValueError("Embedding response indices do not match inputs")
        self.last_tokens=result.get("usage",{}).get("prompt_tokens")
        return normalize([np.frombuffer(base64.b64decode(row["embedding"]), dtype="<f4")
                          if isinstance(row["embedding"], str) else row["embedding"]
                          for row in rows], len(texts), self.config["dimension"])


def normalize(values, count, dimension):
    values=np.ascontiguousarray(values,dtype="float32")
    if values.shape != (count,dimension) or not np.isfinite(values).all():
        raise ValueError("Invalid embedding count/dimension/values")
    if np.any(np.linalg.norm(values,axis=1)==0):
        raise ValueError("Zero embedding")
    faiss.normalize_L2(values)
    return values


class Retriever(VectorRetriever):
    name="api-embedding-exact-ip"
    def __init__(self, settings, encoder=None):
        started=time.monotonic()
        self.settings=settings
        path=resolve_path(settings["vector"]["index_dir"])
        self.meta=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
        if not self.meta["complete"] or self.meta["embedding"] != settings["embedding"]:
            raise ValueError("Incomplete or incompatible vector index")
        from retrieval.chunking import load_meta
        if self.meta["source"] != load_meta(settings["chunks_dir"]):
            raise ValueError("Source chunk metadata changed")
        self.index=faiss.deserialize_index(np.frombuffer((path/"index.faiss").read_bytes(),dtype="uint8"))
        self.chunks=[json.loads(line) for line in (path/"mapping.jsonl").read_text(encoding="utf-8").splitlines()]
        if self.index.ntotal != len(self.chunks) or self.index.d != settings["embedding"]["dimension"]:
            raise ValueError("Index mapping/dimension mismatch")
        self.encoder=encoder or Encoder(settings["embedding"])
        self.last_query_tokens=0
        self.load_seconds=time.monotonic()-started

    def describe(self):
        return {"name":self.name,"model":self.settings["embedding"]["model"],"chunks":self.index.ntotal,"load_seconds":self.load_seconds}

    def search(self, query, top_k):
        self.last_query_tokens=0
        if not query.strip() or top_k<=0 or not self.chunks:return []
        vector=self.encoder.encode([query])
        self.last_query_tokens=self.encoder.last_tokens
        scores,indices=self.index.search(vector,self.index.ntotal)
        ordered=sorted(zip(scores[0],indices[0]),key=lambda row:(-float(row[0]),self.chunks[int(row[1])]["chunk_id"]))
        docs={}
        for chunk_rank,(score,index) in enumerate(ordered,1):
            chunk=self.chunks[int(index)]
            doc_id=chunk["doc_id"]
            if doc_id not in docs:
                rank=len(docs)+1
                hit=dict(chunk,score=float(score),rank=chunk_rank,source="dense")
                docs[doc_id]=dict(doc_id=doc_id,rank=rank,score=float(score),hits=[hit])
                if len(docs)>=top_k:break
        return list(docs.values())


class Fusion(DocumentFusion):
    name="document-rrf"
    def __init__(self, settings):
        self.k=settings["vector"].get("rrf_k",60)
        self.dense_only=settings["vector"].get("dense_only",False)
    def fuse(self,bm25_docs,vector_docs,top_k):
        merged={}
        for source,docs in (("bm25",[] if self.dense_only else bm25_docs),("dense",vector_docs)):
            seen=set()
            for doc in docs:
                doc_id=doc["doc_id"]
                if doc_id in seen:continue
                seen.add(doc_id)
                item=merged.setdefault(doc_id,dict(doc_id=doc_id,score=0.,hits=[],source_ranks={}))
                item["score"]+=1/(self.k+doc["rank"])
                item["hits"].extend(doc["hits"])
                item["source_ranks"][source]=doc["rank"]
        docs=sorted(merged.values(),key=lambda d:(-d["score"],d["doc_id"]))[:min(top_k,10)]
        return [dict(doc,rank=i) for i,doc in enumerate(docs,1)]


def create_retriever(settings):return Retriever(settings)
def create_fusion(settings):return Fusion(settings)
