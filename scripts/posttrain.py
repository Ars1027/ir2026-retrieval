#!/usr/bin/env python3
"""Bonus 2 entry: models, collect, prepare, train, run, compare, freeze and package.

Course QA and decisions stay in gitignored directories. The frozen adapter and
aggregate statistics can be inspected separately from the private course data.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ir2026 import read_jsonl  # noqa: E402
from pipeline.settings import resolve_path  # noqa: E402
from posttraining.config import comparison_signature, load_config  # noqa: E402
from posttraining.data import build_dataset, sha256  # noqa: E402


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def command(parts, env=None, log=None):
    print("+ " + " ".join(map(str, parts)), flush=True)
    if log is None:
        subprocess.run(parts, cwd=ROOT, env=env, check=True)
    else:
        with Path(log).open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(parts, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, parts)


def download_models():
    from huggingface_hub import snapshot_download
    for name in ("student_before", "teacher"):
        model = load_config(f"configs/posttraining/{name}.json")["model"]
        directory = resolve_path(model["path"])
        snapshot_download(model["id"], revision=model["revision"], local_dir=directory,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md"])
        files = {p.name: sha256(p) for p in directory.iterdir()
                 if p.is_file() and p.name != "provenance.json"}
        write_json(directory / "provenance.json", {"id": model["id"], "revision": model["revision"],
                                                  "file_sha256": files})


def run_split(config_path, split, output, limit=None, collect=False):
    config_path, output = resolve_path(config_path), resolve_path(output)
    config = load_config(config_path)
    if collect and split != "train":
        raise ValueError("teacher collection is restricted to train")
    if split == "test":
        if limit is not None or not config.get("frozen"):
            raise ValueError("test requires a frozen config and the full split")
        frozen = config.get("frozen_artifact", {})
        adapter = resolve_path(config["model"]["adapter_path"]) / "adapters.safetensors"
        if frozen.get("adapter_sha256") != sha256(adapter):
            raise ValueError("frozen adapter digest mismatch")
    env = os.environ.copy()
    env["IR2026_POSTTRAIN_CONFIG"] = str(config_path)
    env["IR2026_DECISIONS_DIR"] = str(output / "decisions")
    from posttraining.policy import SYSTEM_PROMPT
    import hashlib
    resources = {"prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                 "bm25_sha256": {p.name: sha256(p) for p in Path(config["bm25_index_dir"]).iterdir() if p.is_file()},
                 "base_weights_sha256": {p.name: sha256(p) for p in resolve_path(config["model"]["path"]).glob("*.safetensors")}}
    label = config["model"]["id"] + "@" + config["model"]["revision"]
    if config["model"].get("adapter_path"):
        label += "+lora:" + sha256(resolve_path(config["model"]["adapter_path"]) / "adapters.safetensors")[:16]
    cmd = [sys.executable, "run.py", "--agent", "agent.posttrained_solution",
           "--method-type", "agent", "--method-label", config["method"],
           "--model-label", label, "--split", split, "--output", str(output)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    started = time.monotonic()
    command(cmd, env)
    write_json(output / "method_config.json", config)
    info_path = output / "run_info.json"
    info = json.loads(info_path.read_text())
    info["comparison_signature"] = comparison_signature(config)
    info["resources"] = resources
    info["wall_seconds"] = round(time.monotonic() - started, 3)
    write_json(info_path, info)
    check = [sys.executable, "check_results.py", "--split", split, "--run-dir", str(output)]
    if limit is None:
        check += ["--submission"]
    else:
        check += ["--questions", str(output / "questions.jsonl")]
    command(check, log=output / "check.log")
    if split != "test":
        evaluation = [sys.executable, "evaluate.py", "--split", split, "--run-dir", str(output)]
        if limit is not None:
            evaluation.append("--allow-subset")
        command(evaluation, log=output / "evaluate.log")


def prepare(teacher_run, output):
    from transformers import AutoTokenizer
    student = load_config("configs/posttraining/student_before.json")
    tokenizer = AutoTokenizer.from_pretrained(resolve_path(student["model"]["path"]), local_files_only=True)
    run_dir = resolve_path(teacher_run)
    command([sys.executable, "check_results.py", "--split", "train", "--run-dir", str(run_dir), "--submission"])
    stats = build_dataset(ROOT / "datas", run_dir, resolve_path(output), tokenizer)
    write_json(ROOT / "records/posttraining/data_statistics.json", stats)
    print(json.dumps(stats, indent=2))


def train(config_path):
    import yaml
    config_path = resolve_path(config_path)
    settings = yaml.safe_load(config_path.read_text())
    if settings.get("mask_prompt") is not True:
        raise ValueError("mask_prompt=true is mandatory: tool observations are inputs only")
    data_dir = resolve_path(settings["data"])
    stats = json.loads((data_dir / "statistics.json").read_text())
    if stats["source_split"] != "train" or stats["max_sequence_tokens"] > settings["max_seq_length"]:
        raise ValueError("wrong training split or silently truncated sequences")
    for name, digest in stats["file_sha256"].items():
        if sha256(data_dir / name) != digest:
            raise ValueError("SFT data changed after verification")
    adapter = resolve_path(settings["adapter_path"])
    if adapter.exists() and any(adapter.iterdir()):
        raise ValueError("adapter output must be new/empty; training never overwrites existing weights")
    adapter.mkdir(parents=True, exist_ok=True)
    log = adapter / "training.log"
    started = time.monotonic()
    command([sys.executable, "-u", "-m", "mlx_lm", "lora", "--config", str(config_path)], log=log)
    if not (adapter / "adapters.safetensors").is_file():
        raise ValueError("training did not produce an adapter")
    log_text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log.read_text())
    val = [{"iteration": int(i), "loss": float(loss)} for i, loss in
           re.findall(r"^\s*(\d+)\s+val\s+([0-9.]+)", log_text, re.MULTILINE)]
    training_loss = [{"iteration": int(i), "loss": float(loss)} for i, loss in
                     re.findall(r"^\s*(\d+)\s+([0-9.]+)\s+[▼▲]", log_text, re.MULTILINE)]
    if not val or not training_loss or training_loss[-1]["iteration"] != settings["iters"]:
        raise ValueError("cannot verify completed training iterations/losses from the MLX log")
    # MLX evaluates BEFORE the numbered iteration, but saves AFTER it. Do not
    # mislabel the former as the latter's score. Use the fixed final checkpoint.
    selected = {"iteration": settings["iters"], "rule": "fixed final iteration, chosen before dev evaluation"}
    versions = {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "numpy", "transformers")}
    record = {"status": "completed", "training": settings, "base_model": load_config("configs/posttraining/student_before.json")["model"],
              "data": stats, "validation_loss": val, "training_loss": training_loss, "selected_checkpoint": selected,
              "selection_rule": selected["rule"],
              "validation_timing": "MLX validates before each reported iteration (weights at iteration minus one)",
              "wall_seconds": round(time.monotonic() - started, 3), "versions": versions,
              "python": platform.python_version(), "platform": platform.platform(),
              "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "adapter_bytes": (adapter / "adapters.safetensors").stat().st_size,
              "adapter_sha256": sha256(adapter / "adapters.safetensors"),
              "log_sha256": sha256(log), "config_sha256": sha256(config_path),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                                 sorted((ROOT / "posttraining").glob("*.py"))}}
    write_json(adapter / "training_summary.json", record)
    write_json(ROOT / "records/posttraining/training_summary.json", record)


def compare(before, after, output):
    from posttraining.report import compare_runs
    summary = compare_runs(ROOT, resolve_path(before), resolve_path(after))
    write_json(resolve_path(output), summary)
    print(json.dumps(summary, indent=2))


def freeze(config_path, output):
    config = load_config(config_path)
    adapter = resolve_path(config["model"]["adapter_path"])
    if not (adapter / "training_summary.json").exists():
        raise ValueError("cannot freeze an untrained/unrecorded adapter")
    export = ROOT / "weights/posttraining"
    export.mkdir(parents=True, exist_ok=True)
    shutil.copy2(adapter / "adapters.safetensors", export / "adapters.safetensors")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    write_json(export / "adapter_config.json", {key: adapter_config[key] for key in
                                                ("fine_tune_type", "num_layers", "lora_parameters")})
    config["model"]["adapter_path"] = "weights/posttraining"
    config["frozen"] = True
    config["frozen_artifact"] = {"adapter_sha256": sha256(adapter / "adapters.safetensors"),
                                  "training_summary_sha256": sha256(adapter / "training_summary.json")}
    config["bm25_index_dir"] = "artifacts/bm25"
    write_json(resolve_path(output), config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("models")
    commands.add_parser("plot")
    collect = commands.add_parser("collect")
    collect.add_argument("--output", required=True)
    collect.add_argument("--limit", type=int, help="train-only smoke; incomplete runs cannot form an SFT dataset")
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--teacher-run", required=True)
    prepare_parser.add_argument("--output", default="artifacts/posttraining/data")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", default="configs/posttraining/lora.yaml")
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--split", choices=["train", "dev", "test"], required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--limit", type=int)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--before", required=True)
    comparison.add_argument("--after", required=True)
    comparison.add_argument("--output", default="records/posttraining/dev_comparison.json")
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--config", default="configs/posttraining/student_after.json")
    frozen.add_argument("--output", default="configs/posttraining/student_frozen.json")
    package = commands.add_parser("package")
    package.add_argument("--before", required=True)
    package.add_argument("--after", required=True)
    package.add_argument("--test", required=True)
    package.add_argument("--teacher", required=True)
    package.add_argument("--output", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "models":
        download_models()
    elif args.action == "plot":
        from posttraining.plot import plot_records
        plot_records(ROOT)
    elif args.action == "collect":
        run_split("configs/posttraining/teacher.json", "train", args.output, args.limit, collect=True)
    elif args.action == "prepare":
        prepare(args.teacher_run, args.output)
    elif args.action == "train":
        train(args.config)
    elif args.action == "run":
        run_split(args.config, args.split, args.output, args.limit)
    elif args.action == "compare":
        compare(args.before, args.after, args.output)
    elif args.action == "freeze":
        freeze(args.config, args.output)
    elif args.action == "package":
        from posttraining.delivery import make_delivery
        result = make_delivery(ROOT, resolve_path(args.before), resolve_path(args.after),
                               resolve_path(args.test), resolve_path(args.teacher), resolve_path(args.output))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
