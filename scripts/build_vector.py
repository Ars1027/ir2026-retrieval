"""Build resumable embedding batches from member A's complete chunks."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from http.client import RemoteDisconnected, IncompleteRead
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from pipeline.settings import load_settings, resolve_path
from retrieval.chunking import iter_chunks_file, load_meta
from vector_backend.ada import Encoder


def encode_retry(encoder, texts):
    for attempt in range(4):
        try:
            return encoder.encode(texts)
        except HTTPError as exc:
            if exc.code == 429 and len(texts) > 8:
                vectors = []
                tokens = 0
                for offset in range(0, len(texts), 8):
                    vectors.append(encode_retry(encoder, texts[offset:offset + 8]))
                    used = encoder.last_tokens
                    tokens = tokens + used if tokens is not None and used is not None else None
                encoder.last_tokens = tokens
                return np.concatenate(vectors, axis=0)
            if attempt == 3 or (exc.code != 429 and not 500 <= exc.code < 600):
                raise
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
            else:
                delay = 60 * (attempt + 1) if exc.code == 429 else 2 ** attempt
            time.sleep(max(0, delay))
        except (URLError, RemoteDisconnected, IncompleteRead, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def cached_batch(number, rows, batches, embedding, encoder=None):
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    saved=batches/f"{number:06d}.npz"
    if saved.exists():
        with np.load(saved,allow_pickle=False) as data:
            if str(data["digest"])!=digest:raise ValueError("Chunk content changed")
            return data["vectors"],int(data["tokens"]),float(data["seconds"])
    # Each worker owns its token counter; no shared mutable API client.
    client=encoder or Encoder(embedding)
    started=time.monotonic()
    vectors=encode_retry(client,[r["text"] for r in rows])
    seconds=time.monotonic()-started
    used=client.last_tokens if client.last_tokens is not None else -1
    temporary=saved.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream,vectors=vectors,tokens=used,seconds=seconds,digest=digest)
    temporary.replace(saved)
    return vectors,used,seconds


def build(settings, encoder=None, workers=8):
    if workers < 1:raise ValueError("workers must be positive")
    target=resolve_path(settings["vector"]["index_dir"])
    target.mkdir(parents=True,exist_ok=True)
    source=load_meta(settings["chunks_dir"])
    identity={"embedding":settings["embedding"],"source":source}
    manifest=target/"metadata.json"
    if manifest.exists():
        prior=json.loads(manifest.read_text())
        if any(prior[k]!=v for k,v in identity.items()):raise ValueError("Index configuration/source mismatch; use new directory")
        if prior["complete"]:return prior
    manifest.write_text(json.dumps(dict(identity,complete=False),indent=2))
    batches=target/"batches";batches.mkdir(exist_ok=True)
    chunks=iter_chunks_file(settings["chunks_dir"])
    index=faiss.IndexFlatIP(settings["embedding"]["dimension"])
    tokens=0;elapsed=0.;count=0
    mapping=target/"mapping.jsonl.tmp"
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor, mapping.open("w",encoding="utf-8") as output:
        pending=deque()
        number=0
        exhausted=False
        while pending or not exhausted:
            while not exhausted and len(pending)<workers * 16:
                rows=list(itertools.islice(chunks,settings["embedding"]["batch_size"]))
                if not rows:
                    exhausted=True
                    break
                future=executor.submit(cached_batch,number,rows,batches,settings["embedding"],encoder)
                pending.append((rows,future))
                number+=1
            if not pending:break
            rows,future=pending.popleft()
            vectors,used,seconds=future.result()
            index.add(vectors)
            tokens=None if tokens is None or used<0 else tokens+used
            elapsed+=seconds;count+=len(rows)
            for row in rows:output.write(json.dumps({k:row[k] for k in ("chunk_id","doc_id","start","end")},ensure_ascii=False)+"\n")
            print(f"encoded {count}/{source['chunks']}",flush=True)
    if count!=source["chunks"]:raise ValueError("Source chunk count mismatch")
    (target/"index.faiss").write_bytes(faiss.serialize_index(index).tobytes())
    mapping.replace(target/"mapping.jsonl")
    result=dict(identity,complete=True,count=count,input_tokens=tokens,encoding_seconds=elapsed,
        build_wall_seconds=time.monotonic()-started,workers=workers,
        index_bytes=(target/"index.faiss").stat().st_size,model_revision_note="Provider model alias; immutable revision unavailable")
    manifest.write_text(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/vector.ada.json")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args();print(json.dumps(build(load_settings(args.config),workers=args.workers),indent=2))
