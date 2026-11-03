"""Minimal OpenAI-compatible chat client used only for optional answer generation.

The key is read from the environment; it is never written to configs, logs or
traces. When no key is configured the client reports itself unavailable and the
pipeline degrades to ``no_answer`` instead of failing the run.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.openai.com/v1"
KEY_ENV = "IR2026_MODEL_API_KEY"


class ModelClientError(RuntimeError):
    pass


class ModelClient:
    def __init__(self, settings=None):
        config = ((settings or {}).get("answer") or {})
        self.model = config.get("model")
        self.base_url = (config.get("base_url")
                         or os.environ.get("IR2026_MODEL_BASE_URL")
                         or os.environ.get("OPENAI_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(config.get("timeout_seconds") or 30)
        self.max_output_tokens = int(config.get("max_output_tokens") or 200)
        self.key_env = config.get("api_key_env") or KEY_ENV
        self.api_key = os.environ.get(self.key_env) or os.environ.get("OPENAI_API_KEY")
        self.last_usage = None

    def available(self) -> bool:
        return bool(self.model and self.api_key)

    def describe(self) -> dict:
        return {"model": self.model, "base_url": self.base_url,
                "timeout_seconds": self.timeout, "max_output_tokens": self.max_output_tokens,
                "key_present": bool(self.api_key), "key_env": self.key_env}

    def complete(self, system: str, user: str) -> tuple:
        """Return ``(text, usage)``; raises ModelClientError on any failure."""
        if not self.model:
            raise ModelClientError("no answer model configured")
        if not self.api_key:
            raise ModelClientError(f"missing API key in environment variable {self.key_env}")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": self.max_output_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise ModelClientError(f"HTTP {exc.code} from model endpoint: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelClientError(f"model request failed: {exc}") from exc
        except ValueError as exc:
            raise ModelClientError(f"invalid model response: {exc}") from exc
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelClientError("model response has no message content") from exc
        usage = body.get("usage") or {}
        self.last_usage = {"model_input_tokens": usage.get("prompt_tokens"),
                           "model_output_tokens": usage.get("completion_tokens"),
                           "token_notes": "Reported by the chat completion API usage field."}
        return (text or "").strip(), dict(self.last_usage)
