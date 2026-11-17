"""Build next-action examples from TRAIN trajectories, grouped by question.

Only this offline preprocessing module reads train labels. The inference agent
does not import it. No dev/test labels or upstream answers are opened.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ir2026 import keyed, load_questions, normalize_answer, read_jsonl, write_jsonl
from posttraining.policy import SYSTEM_PROMPT, encode, messages_for, parse_action


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_ids(data_dir):
    """The training path never opens dev/test questions or labels.

    The official verify_data.py checks the whole course release separately.
    """
    questions = load_questions(Path(data_dir) / "train.jsonl")
    if len({normalize_answer(q["question"]) for q in questions}) != len(questions):
        raise ValueError("duplicate training question texts")
    return keyed(questions, "train")


def visible_gold_answer(messages, label, final_doc_ids):
    """Conservatively require the train answer in an actual visible gold passage.

    Initial questions, action strings, IDs and error messages cannot establish
    support. Only text fields inside real TOOL_RESULT observations qualify.
    """
    answer = normalize_answer(label["answer"])
    if not answer:
        return False
    pattern = re.compile(r"(?<!\w)" + re.escape(answer) + r"(?!\w)")
    gold = set(label["gold_doc_ids"]) & set(final_doc_ids)
    for message in messages:
        content = message.get("content", "")
        if message.get("role") != "user" or not content.startswith("TOOL_RESULT "):
            continue
        observation = json.loads(content[len("TOOL_RESULT "):])
        passages = observation.get("passages", [observation])
        for passage in passages:
            if passage.get("doc_id") in gold and pattern.search(normalize_answer(passage.get("text", ""))):
                return True
    return False


def build_dataset(data_dir, run_dir, out_dir, tokenizer, max_tokens=4096, seed=2026):
    data_dir, run_dir, out_dir = map(Path, (data_dir, run_dir, out_dir))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("use a new/empty SFT data directory")
    info = json.loads((run_dir / "run_info.json").read_text())
    if info.get("split") != "train" or info.get("method_type") != "agent":
        raise ValueError("teacher trajectories must be collected on train in agent mode")
    train = training_ids(data_dir)
    labels = keyed(read_jsonl(data_dir / "train_labels.jsonl"), "train labels")
    results = keyed(read_jsonl(run_dir / "results.jsonl"), "teacher results")
    if set(results) != set(train):
        raise ValueError("collect the complete training split before preparing SFT data")
    # Split BEFORE filtering or expanding into turns: a question never spans both.
    ordered = sorted(train, key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    n_valid = max(1, len(ordered) // 10)
    valid_ids = set(ordered[:n_valid])
    examples = {"train": [], "valid": []}
    provenance, reasons, actions, target_sources = [], Counter(), Counter(), Counter()
    accepted_questions, lengths, supervised = Counter(), [], []
    for qid in ordered:
        result = results[qid]
        label = labels[qid]
        if result["status"] != "ok":
            reasons["trajectory_not_ok"] += 1
            continue
        if not set(result["doc_ids"][:10]) & set(label["gold_doc_ids"]):
            reasons["trajectory_misses_gold_document"] += 1
            continue
        decisions = read_jsonl(run_dir / "decisions" / f"{qid}.jsonl")
        split = "valid" if qid in valid_ids else "train"
        count = 0
        for decision in decisions:
            if not decision.get("action_valid"):
                reasons["invalid_action"] += 1
                continue
            action = parse_action(decision["output"])
            messages = decision["messages"]
            if (not messages or messages[0] != {"role": "system", "content": SYSTEM_PROMPT}
                    or messages[1] != messages_for(train[qid]["question"], [])[1]
                    or messages[-1]["role"] != "user"):
                raise ValueError("trajectory prompt differs from the runtime policy")
            target, source = decision["output"], "teacher_action"
            if action["action"] == "finish":
                if len(label["answer"]) > 200 or not visible_gold_answer(messages, label, action["doc_ids"]):
                    reasons["finish_without_visible_gold_answer"] += 1
                    continue
                if normalize_answer(action["answer"]) != normalize_answer(label["answer"]):
                    # A separate, explicitly labelled train supervision source.
                    # Never rewrite the original teacher decisions or tool trace.
                    target = encode({**action, "answer": label["answer"]})
                    source = "visible_gold_train_label_correction"
            full = messages + [{"role": "assistant", "content": target}]
            tokens = tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False,
                                                    return_dict=False)
            prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=False)
            if tokens[:len(prefix)] != prefix or len(tokens) <= len(prefix):
                raise ValueError("chat template does not support exact assistant-only loss masking")
            if len(tokens) > max_tokens:
                reasons["overlength_turn"] += 1
                continue
            examples[split].append({"messages": full})
            provenance.append({"split": split, "question_id": qid, "turn": decision["turn"],
                               "tokens": len(tokens), "loss_start": len(prefix),
                               "action": action["action"], "target_source": source,
                               "original_teacher_output_sha256": hashlib.sha256(decision["output"].encode()).hexdigest()})
            lengths.append(len(tokens))
            supervised.append(len(tokens) - len(prefix))
            actions[action["action"]] += 1
            target_sources[source] += 1
            count += 1
        if count:
            accepted_questions[split] += 1
    if not examples["train"] or not examples["valid"]:
        raise ValueError("quality filtering left an empty train/valid set; inspect teacher runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in examples.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    write_jsonl(out_dir / "provenance.jsonl", provenance)
    stats = {"source_split": "train", "source_questions": len(train), "seed": seed,
             "internal_validation_questions_before_filter": n_valid,
             "accepted_questions": dict(accepted_questions),
             "examples": {k: len(v) for k, v in examples.items()},
             "action_counts": dict(actions), "filtered": dict(reasons),
             "target_sources": dict(target_sources),
             "max_sequence_tokens": max(lengths), "supervised_tokens": sum(supervised),
             "mask_prompt": True, "tool_observations_are_targets": False,
             "quality_filter": "trajectory status=ok and train GoldHit@10=1; finish requires the train answer in a visible final gold passage; wrong final answers are explicitly corrected from train labels",
             "file_sha256": {name: sha256(out_dir / name) for name in ("train.jsonl", "valid.jsonl")},
             "source_sha256": {name: sha256(data_dir / name) for name in
                                ("train.jsonl", "train_labels.jsonl")}}
    (out_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def rebalance_dataset(source_dir, out_dir, finish_multiplier=3):
    """Repeat terminal TRAIN samples without mixing groups or changing targets."""
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    if type(finish_multiplier) is not int or finish_multiplier < 1:
        raise ValueError("finish multiplier must be a positive integer")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("use a new/empty balanced data directory")
    stats = json.loads((source_dir / "statistics.json").read_text())
    for name, digest in stats["file_sha256"].items():
        if sha256(source_dir / name) != digest:
            raise ValueError("source SFT data digest mismatch")
    provenance = read_jsonl(source_dir / "provenance.jsonl")
    expanded, metadata = {}, []
    for split in ("train", "valid"):
        rows = read_jsonl(source_dir / f"{split}.jsonl")
        origins = [r for r in provenance if r["split"] == split]
        expanded[split] = []
        for row, origin in zip(rows, origins, strict=True):
            repeats = finish_multiplier if split == "train" and origin["action"] == "finish" else 1
            for replica in range(repeats):
                expanded[split].append(row)
                metadata.append({**origin, "replica": replica})
    train_ids = {r["question_id"] for r in metadata if r["split"] == "train"}
    valid_ids = {r["question_id"] for r in metadata if r["split"] == "valid"}
    if train_ids & valid_ids:
        raise ValueError("balanced data mixed question groups")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in expanded.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    write_jsonl(out_dir / "provenance.jsonl", metadata)
    stats = {**stats, "unique_examples": stats["examples"],
             "unique_action_counts": stats["action_counts"],
             "unique_target_sources": stats["target_sources"],
             "source_dataset_sha256": stats["file_sha256"],
             "sampling": {"finish_multiplier": finish_multiplier, "applied_to": "train only"},
             "examples": {split: len(rows) for split, rows in expanded.items()},
             "action_counts": dict(Counter(r["action"] for r in metadata)),
             "target_sources": dict(Counter(r["target_source"] for r in metadata)),
             "supervised_tokens": sum(r["tokens"] - r["loss_start"] for r in metadata),
             "file_sha256": {name: sha256(out_dir / name) for name in ("train.jsonl", "valid.jsonl")}}
    (out_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats
