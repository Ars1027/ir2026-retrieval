"""Exact normalized inner-product index with persistent original chunk IDs."""
import hashlib
import json
from pathlib import Path
import time
import faiss
import numpy as np
from retrieval.embedding import QwenEncoder

FORMAT_VERSION = 1

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def normalized(vectors, count, dimension):
    vectors = np.ascontiguousarray(vectors, dtype='float32')
    if vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        raise ValueError('Embedding dimension/count or finite-value mismatch')
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError('Zero embedding')
    faiss.normalize_L2(vectors)
    return vectors

def build_index(chunks, metadata, index_dir, settings, encoder=None):
    target = Path(index_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Index directory is not empty: {target}')
    started = time.perf_counter()
    embedding = settings['embedding']
    encoder = encoder or QwenEncoder(embedding)
    index = faiss.IndexFlatIP(embedding['dimension'])
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Duplicate chunk_id')
    batch = embedding['batch_size']
    for offset in range(0, len(chunks), batch):
        rows = chunks[offset:offset + batch]
        index.add(normalized(encoder.encode_documents([c['text'] for c in rows]), len(rows), index.d))
    mapping = [{k: c[k] for k in ('chunk_id','doc_id','start','end')} for c in chunks]
    manifest = {'format_version': FORMAT_VERSION, 'embedding': embedding, 'source_metadata': metadata,
        'chunks_sha256': fingerprint(chunks), 'count': len(chunks), 'build_seconds': time.perf_counter()-started}
    target.mkdir(parents=True, exist_ok=True)
    (target/'index.faiss').write_bytes(faiss.serialize_index(index).tobytes())
    (target/'chunks.json').write_text(json.dumps(mapping, ensure_ascii=False), encoding='utf-8')
    (target/'metadata.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

class DenseRetriever:
    def __init__(self, index_dir, settings, encoder=None):
        started = time.perf_counter()
        path = Path(index_dir)
        self.metadata = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        if self.metadata['format_version'] != FORMAT_VERSION or self.metadata['embedding'] != settings['embedding']:
            raise ValueError('Dense index version/model configuration mismatch; rebuild index')
        chunks_dir = settings.get('chunks_dir')
        if chunks_dir:
            source = Path(chunks_dir)/'metadata.json'
            if not source.exists() or json.loads(source.read_text(encoding='utf-8')) != self.metadata['source_metadata']:
                raise ValueError('Source chunk metadata mismatch; rebuild index')
        self.chunks = json.loads((path/'chunks.json').read_text(encoding='utf-8'))
        self.index = faiss.deserialize_index(np.frombuffer((path/'index.faiss').read_bytes(), dtype='uint8'))
        if self.index.d != settings['embedding']['dimension'] or self.index.ntotal != len(self.chunks) or len(self.chunks) != self.metadata['count']:
            raise ValueError('Dense index dimension/mapping mismatch')
        self.encoder = encoder or QwenEncoder(settings['embedding'])
        self.load_seconds = time.perf_counter()-started
        self.last_query_tokens = 0

    def search(self, query, top_k):
        if not query.strip() or top_k <= 0 or not self.chunks:
            self.last_query_tokens = 0
            return []
        vector = normalized(self.encoder.encode_query(query), 1, self.index.d)
        self.last_query_tokens = getattr(self.encoder, 'last_tokens', None)
        # Fetch all exact scores so ties at the cut-off are deterministic.
        scores, ids = self.index.search(vector, self.index.ntotal)
        rows = sorted(zip(scores[0], ids[0]), key=lambda x: (-float(x[0]), self.chunks[int(x[1])]['chunk_id']))[:top_k]
        return [dict(self.chunks[int(i)], score=float(score), rank=rank, source='dense')
                for rank,(score,i) in enumerate(rows,1)]
