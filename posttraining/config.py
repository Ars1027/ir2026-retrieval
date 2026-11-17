"""Posttraining configs are separate from the frozen basic-task configs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pipeline.settings import resolve_path


def load_config(path=None):
    path = resolve_path(path or os.environ["IR2026_POSTTRAIN_CONFIG"])
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("method") or not isinstance(config.get("model"), dict):
        raise ValueError("posttraining config needs method and model")
    model = config["model"]
    if not model.get("id") or not model.get("revision") or not model.get("path"):
        raise ValueError("pin the model ID, revision and local path")
    policy = config.get("policy", {})
    if not 1 <= policy.get("max_tool_calls", 10) <= 32 or not 1 <= policy.get("max_turns", 12) <= 33:
        raise ValueError("invalid policy budget")
    config["bm25_index_dir"] = str(resolve_path(config["bm25_index_dir"]))
    return config


def comparison_signature(config):
    """Everything except method label/adapter must match for a paired comparison."""
    relevant = {k: v for k, v in config.items() if k not in {"method", "frozen"}}
    relevant["model"] = {k: v for k, v in config["model"].items() if k != "adapter_path"}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
