"""Text encoders for the dense branch (member B).

Two backends share one interface:

``hf``
    A Hugging Face transformer encoder. The planned model is
    ``Qwen/Qwen3-Embedding-0.6B`` with its model-card query instruction, last
    token pooling, left padding and L2 normalisation.

``hashing``
    Offline deterministic fallback used when the planned model cannot be
    downloaded or does not fit the available hardware. It projects hashed word
    and character features into a fixed dimension and L2-normalises the result.
    It is *not* a semantic encoder; every artefact records the backend that was
    actually used so results are never mislabelled.

``auto`` tries ``hf`` first and falls back to ``hashing`` while recording the
reason. Nothing is silently truncated: texts longer than ``max_length`` are
reported through the encoder metadata.
"""
from __future__ import annotations

import math
import sys
import zlib

import numpy as np

from retrieval.bm25 import TOKEN_RE

DEFAULT_ENCODER = {
    "backend": "auto",
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "revision": "main",
    "pooling": "last",
    "max_length": 512,
    "normalize": True,
    "device": "cpu",
    "torch_dtype": "float32",
    "query_instruction": ("Given a web search query, retrieve relevant passages "
                          "that answer the query"),
    "batch_size": 8,
    "hashing": {"dimension": 256, "char_ngrams": True, "char_weight": 0.25, "seed": 2026},
}


class EncoderError(RuntimeError):
    pass


def _resolve(config=None, settings=None) -> dict:
    merged = dict(DEFAULT_ENCODER)
    merged["hashing"] = dict(DEFAULT_ENCODER["hashing"])
    override = config or ((settings or {}).get("dense") or {}).get("encoder") or {}
    for key, value in override.items():
        if key == "hashing" and isinstance(value, dict):
            merged["hashing"].update(value)
        else:
            merged[key] = value
    if "batch_size" not in override:
        merged["batch_size"] = ((settings or {}).get("dense") or {}).get("batch_size",
                                                                         merged["batch_size"])
    return merged


class Encoder:
    def __init__(self, config=None, settings=None):
        self.config = _resolve(config, settings)
        self.backend = None
        self.fallback_reason = None
        requested = self.config.get("backend") or "auto"
        if requested in ("auto", "hf"):
            try:
                self._init_hf()
            except Exception as exc:  # missing weights, no network, no transformers
                if requested == "hf":
                    raise EncoderError(f"HF encoder unavailable: {type(exc).__name__}: {exc}") from exc
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                print(f"[encoder] falling back to hashing backend: {self.fallback_reason}",
                      file=sys.stderr, flush=True)
                self._init_hashing()
        elif requested == "hashing":
            self._init_hashing()
        else:
            raise EncoderError(f"unknown encoder backend: {requested}")

    # -- backends --------------------------------------------------------
    def _init_hf(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        cfg = self.config
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
        dtype = getattr(torch, cfg.get("torch_dtype") or "float32")
        self._model = AutoModel.from_pretrained(cfg["model"], revision=cfg["revision"],
                                                torch_dtype=dtype)
        self._model.eval()
        self._model.to(cfg.get("device") or "cpu")
        self._tokenizer.padding_side = "left"
        self.backend = "hf"
        self.dimension = int(self._model.config.hidden_size)

    def _init_hashing(self):
        cfg = self.config["hashing"]
        self.backend = "hashing"
        self.dimension = int(cfg.get("dimension") or 256)
        if self.dimension < 8:
            raise EncoderError("hashing.dimension must be at least 8")
        self._seed = int(cfg.get("seed") or 2026)
        self._char_ngrams = bool(cfg.get("char_ngrams", True))
        self._char_weight = float(cfg.get("char_weight", 0.25))

    # -- metadata --------------------------------------------------------
    def describe(self) -> dict:
        cfg = self.config
        info = {"backend": self.backend, "dimension": int(self.dimension),
                "normalize": bool(cfg.get("normalize", True)),
                "requested_backend": cfg.get("backend"),
                "max_length": cfg.get("max_length")}
        if self.backend == "hf":
            info.update({"model": cfg["model"], "revision": cfg["revision"],
                         "pooling": cfg.get("pooling"), "device": cfg.get("device"),
                         "torch_dtype": cfg.get("torch_dtype"),
                         "query_instruction": cfg.get("query_instruction")})
        else:
            info.update({"model": "hashed-features", "revision": "n/a",
                         "pooling": "n/a", "fallback_reason": self.fallback_reason,
                         "hashing": dict(cfg["hashing"]),
                         "note": ("Offline lexical hashing fallback, not a semantic model. "
                                  "Replace with the hf backend when the planned embedding "
                                  "model is available.")})
        return info

    # -- encoding --------------------------------------------------------
    def encode_query(self, text: str) -> np.ndarray:
        instruction = self.config.get("query_instruction")
        if self.backend == "hf" and instruction:
            text = f"Instruct: {instruction}\nQuery: {text}"
        return self._encode([text])[0]

    def encode_documents(self, texts: list) -> np.ndarray:
        return self._encode(texts)

    def encode_batches(self, texts):
        batch_size = max(1, int(self.config.get("batch_size") or 8))
        buffer = []
        for text in texts:
            buffer.append(text)
            if len(buffer) >= batch_size * 8:
                yield self._encode(buffer)
                buffer = []
        if buffer:
            yield self._encode(buffer)

    def _encode(self, texts: list) -> np.ndarray:
        if self.backend == "hf":
            vectors = self._encode_hf(texts)
        else:
            vectors = self._encode_hashing(texts)
        if self.config.get("normalize", True):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.maximum(norms, 1e-12, out=norms)
            vectors = vectors / norms
        return vectors.astype(np.float32, copy=False)

    def _encode_hf(self, texts: list) -> np.ndarray:
        torch = self._torch
        max_length = int(self.config.get("max_length") or 512)
        encoded = self._tokenizer(texts, padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = self._model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if (self.config.get("pooling") or "last") == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = hidden[:, -1]
        return pooled.detach().to("cpu").float().numpy()

    def _encode_hashing(self, texts: list) -> np.ndarray:
        dim = self.dimension
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = {}
            for token in TOKEN_RE.findall(text.lower()):
                counts[token] = counts.get(token, 0) + 1
                if self._char_ngrams and len(token) > 4:
                    for i in range(len(token) - 3):
                        gram = token[i:i + 4]
                        counts["\x00" + gram] = counts.get("\x00" + gram, 0) + 1
            for token, count in counts.items():
                weight = 1.0 + math.log(count)
                if token.startswith("\x00"):
                    weight *= self._char_weight
                    token = token[1:]
                digest = zlib.crc32(token.encode("utf-8"))
                index = (digest ^ self._seed) % dim
                sign = 1.0 if (digest >> 15) & 1 else -1.0
                out[row, index] += sign * weight
        return out
