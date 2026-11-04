"""Method configuration loading.

``IR2026_METHOD_CONFIG`` names the JSON file used by the current run; when it is
unset the BM25 configuration is used. Relative paths inside the file are
resolved against the project root, so a config works from any working directory.

Extension keys of the optional vector branch (``vector``, ``vector_index_dir``)
are preserved verbatim; this loader never drops unknown keys.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VAR = "IR2026_METHOD_CONFIG"
DEFAULT_CONFIG = "configs/final.json"

# Documented defaults; every key can be overridden in the JSON file.
DEFAULTS = {
    "method": "bm25-v1",
    "chunks_dir": "artifacts/chunks",
    "bm25_index_dir": "artifacts/bm25",
    "vector_index_dir": "artifacts/vector",
    "candidate_k": 200,
    "chunking": {"size": 1200, "overlap": 200, "min_chars": 120},
    "bm25": {"k1": 0.9, "b": 0.4, "use_stopwords": True, "capitalized_boost": 1.25},
    "vector": {"enabled": False, "retriever": None, "fusion": None, "top_k": 200},
    "retrieval": {"final_docs": 10, "evidence_docs": 5},
    "answer": {"mode": "extractive", "model": None, "base_url": None,
               "timeout_seconds": 30, "max_output_tokens": 200, "max_answer_chars": 200},
}

PATH_KEYS = ("chunks_dir", "bm25_index_dir", "vector_index_dir")


class SettingsError(ValueError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def config_path(explicit=None) -> Path:
    root = project_root()
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_absolute() else (root / path)
    env = os.environ.get(ENV_VAR)
    if env:
        path = Path(env)
        return path if path.is_absolute() else (root / path)
    return root / DEFAULT_CONFIG


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root() / path)


def load_settings(explicit=None) -> dict:
    """Read the active method config and return a resolved settings dictionary."""
    path = config_path(explicit)
    if not path.is_file():
        raise SettingsError(f"method config not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SettingsError(f"cannot read method config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SettingsError(f"method config must be a JSON object: {path}")
    settings = _merge(DEFAULTS, raw)
    settings["config_path"] = str(path)
    settings["config_digest"] = _digest(raw)
    for key in PATH_KEYS:
        if key in settings:
            settings[key] = str(resolve_path(settings[key]))
    _validate(settings)
    return settings


def _digest(raw: dict) -> str:
    import hashlib
    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validate(settings: dict) -> None:
    for key in ("candidate_k",):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise SettingsError(f"{key} must be a positive integer")
    retrieval = settings.get("retrieval") or {}
    if type(retrieval.get("final_docs")) is not int or not 1 <= retrieval["final_docs"] <= 10:
        raise SettingsError("retrieval.final_docs must be in [1, 10]")
    quotes = retrieval.get("evidence_docs")
    if type(quotes) is not int or not 1 <= quotes <= 5:
        raise SettingsError("retrieval.evidence_docs must be in [1, 5]")
    mode = (settings.get("answer") or {}).get("mode")
    if mode not in ("extractive", "model", "off"):
        raise SettingsError("answer.mode must be extractive, model or off")
