"""Pinned Qwen encoder; refuses truncation and only loads local weights."""
import numpy as np

class QwenEncoder:
    def __init__(self, settings):
        from sentence_transformers import SentenceTransformer
        self.settings = settings
        self.model = SentenceTransformer(settings['model'], revision=settings['revision'],
            device=settings['device'], local_files_only=True)
        self.model.max_seq_length = settings['max_length']
        self.dimension = settings['dimension']
        self.last_tokens = 0

    def _encode(self, texts, query=False):
        prompt = ('Instruct: ' + self.settings['query_instruction'] + '\nQuery:') if query else ''
        lengths = [len(self.model.tokenizer(prompt + text, truncation=False)['input_ids']) for text in texts]
        if any(n > self.settings['max_length'] for n in lengths):
            raise ValueError('Embedding input exceeds max_length; rebuild smaller chunks, never truncate')
        self.last_tokens = sum(lengths)
        return np.asarray(self.model.encode(texts, prompt=prompt, batch_size=self.settings['batch_size'],
            normalize_embeddings=True, convert_to_numpy=True), dtype='float32')

    def encode_documents(self, texts):
        return self._encode(texts)

    def encode_query(self, query):
        return self._encode([query], query=True)
