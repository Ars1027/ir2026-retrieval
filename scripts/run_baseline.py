"""Run a fixed baseline through the unchanged course runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(method, split, config, output, limit=None):
    config = Path(config).resolve()
    output = Path(output).resolve()
    settings = json.loads(config.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory")
    if split == "test" and not settings.get("frozen", False):
        raise ValueError("Test requires a frozen configuration after full dev and cross-review")
    if split == "test" and limit is not None:
        raise ValueError("Final test must cover the full split")
    cmd = [sys.executable, str(ROOT / "run.py"), "--agent", f"agent.{method}_solution",
           "--split", split, "--method-type", "retrieval", "--method-label", f"{method}-v1",
           "--output", str(output)]
    models = []
    if method in ("dense", "hybrid"):
        embedding = settings["embedding"]
        models.append(embedding["model"] + "@" + embedding["revision"])
    answer = settings.get("answer", {})
    if answer.get("model"):
        models.append(answer["model"])
    if models:
        cmd += ["--model-label", ";".join(models)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    env = os.environ.copy()
    env["IR2026_METHOD_CONFIG"] = str(config)
    return cmd, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    config = args.config or ROOT / "configs" / f"{args.method}.json"
    try:
        cmd, env = prepare(args.method, args.split, config, args.output, args.limit)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    code = subprocess.call(cmd, cwd=ROOT, env=env)
    if args.output.exists():
        raw = config.read_bytes()
        (args.output / "method_config.json").write_bytes(raw)
        (args.output / "config_sha256.txt").write_text(hashlib.sha256(raw).hexdigest() + "\n")
    return code

if __name__ == "__main__":
    raise SystemExit(main())
