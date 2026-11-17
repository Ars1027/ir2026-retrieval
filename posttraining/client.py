"""Local, revision-pinned MLX generation; no query or corpus leaves the Mac."""
from __future__ import annotations

import time

from ir2026 import BudgetExceeded


class MLXClient:
    def __init__(self, config):
        from mlx_lm import load
        from pipeline.settings import resolve_path

        self.config = config
        path = str(resolve_path(config["path"]))
        adapter = config.get("adapter_path")
        adapter = str(resolve_path(adapter)) if adapter else None
        self.model, self.tokenizer = load(path, adapter_path=adapter)

    def complete(self, messages, deadline):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                     add_generation_prompt=True)
        if len(prompt) > self.config.get("max_input_tokens", 8192):
            raise ValueError("model context budget exceeded; no silent truncation")
        parts, last = [], None
        for response in stream_generate(self.model, self.tokenizer, prompt=prompt,
                                        max_tokens=self.config.get("max_output_tokens", 256),
                                        sampler=make_sampler(temp=0.0)):
            if time.monotonic() > deadline:
                raise BudgetExceeded("time during model generation")
            parts.append(response.text)
            last = response
        return "".join(parts), {"input": len(prompt),
                                "output": last.generation_tokens if last else 0}
