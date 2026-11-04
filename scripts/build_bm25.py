#!/usr/bin/env python3
"""Build the BM25 index from the chunk store (core).

Usage:
    python scripts/build_bm25.py
    python scripts/build_bm25.py --chunks-dir artifacts/chunks --k1 0.9 --b 0.4
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
from retrieval import bm25  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--chunks-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--k1", type=float)
    parser.add_argument("--b", type=float)
    parser.add_argument("--limit", type=int, help="only the first N chunks (smoke tests)")
    parser.add_argument("--stats-out", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.config)
    params = dict(settings.get("bm25") or {})
    if args.k1 is not None:
        params["k1"] = args.k1
    if args.b is not None:
        params["b"] = args.b
    chunks_dir = args.chunks_dir or Path(settings["chunks_dir"])
    out_dir = args.out_dir or Path(settings["bm25_index_dir"])

    started = time.monotonic()
    stats = bm25.build_index(chunks_dir, out_dir, params, limit=args.limit,
                             progress=lambda n: print(f"  {n} chunks", flush=True))
    stats["elapsed_seconds"] = round(time.monotonic() - started, 3)
    stats["index_bytes"] = bm25.index_size(out_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
