#!/usr/bin/env python3
"""Build faithful chunks for the course corpus (member A).

Usage:
    python scripts/build_chunks.py
    python scripts/build_chunks.py --out-dir artifacts/member_a/chunks --limit 50

The builder reads corpus text only. Questions and labels are never opened.
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
from retrieval import chunking  # noqa: E402
from retrieval.corpus_adapter import CorpusAdapter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="method config JSON (default: IR2026_METHOD_CONFIG or configs/bm25.json)")
    parser.add_argument("--out-dir", type=Path, help="chunk output directory")
    parser.add_argument("--corpus", type=Path, help="corpus root (default: <project>/corpus)")
    parser.add_argument("--size", type=int, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, help="overlap in characters")
    parser.add_argument("--limit", type=int, help="only the first N documents (smoke tests)")
    parser.add_argument("--verify", action="store_true", help="re-read chunks and verify offsets")
    parser.add_argument("--verify-sample", type=int, default=2000)
    parser.add_argument("--stats-out", type=Path, help="write build statistics as JSON")
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.size is not None:
        settings.setdefault("chunking", {})["size"] = args.size
    if args.overlap is not None:
        settings.setdefault("chunking", {})["overlap"] = args.overlap
    out_dir = args.out_dir or (ROOT / "artifacts/member_a/chunks")

    started = time.monotonic()
    stats = chunking.write_chunks(
        corpus_root=args.corpus or (ROOT / "corpus"),
        out_dir=out_dir,
        settings=settings,
        limit=args.limit,
        progress=lambda done, total, doc_id: print(f"  {done}/{total} {doc_id}", flush=True),
    )
    stats["elapsed_seconds"] = round(time.monotonic() - started, 3)
    stats["bytes"] = sum(p.stat().st_size for p in Path(out_dir).glob("*") if p.is_file())
    if args.verify:
        adapter = CorpusAdapter(args.corpus or (ROOT / "corpus"))
        stats["verify"] = chunking.check_faithful(
            chunking.iter_chunks_file(out_dir), adapter, sample=args.verify_sample)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    if stats.get("verify") and not stats["verify"]["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
