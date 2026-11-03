#!/usr/bin/env python3
"""Build the dense chunk index (member B).

Examples:
    python scripts/build_dense.py --config configs/dense.json
    python scripts/build_dense.py --backend hashing --limit 2000   # smoke test

The encoder that was actually used (and any fallback reason) is written into
``meta.json`` next to the vectors.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.settings import load_settings  # noqa: E402
from retrieval import dense  # noqa: E402
from retrieval.embedding import Encoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/dense.json"))
    parser.add_argument("--chunks-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--backend", choices=["auto", "hf", "hashing"], help="override encoder backend")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, help="encode only the first N chunks (smoke tests)")
    parser.add_argument("--stats-out", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.config)
    encoder_config = dict((settings.get("dense") or {}).get("encoder") or {})
    if args.backend:
        encoder_config["backend"] = args.backend
    if args.batch_size:
        encoder_config["batch_size"] = args.batch_size
    started = time.monotonic()
    encoder = Encoder(encoder_config)
    print(json.dumps({"encoder": encoder.describe()}, ensure_ascii=False, indent=2), flush=True)
    meta = dense.build_index(
        chunks_path=args.chunks_dir or Path(settings["chunks_dir"]),
        out_dir=args.out_dir or Path(settings["dense_index_dir"]),
        settings={**settings, "dense": {**settings.get("dense", {}), "encoder": encoder_config}},
        limit=args.limit,
        encoder=encoder,
        progress=lambda done, total: print(f"  {done}/{total} chunks", flush=True),
    )
    meta["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
