"""Build next-action examples from TRAIN trajectories, grouped by question.

Only this offline preprocessing module reads train labels. The inference agent
does not import it. No dev/test labels or upstream answers are opened.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ir2026 import keyed, load_questions, normalize_answer, read_jsonl, write_jsonl
from posttraining.policy import SYSTEM_PROMPT, messages_for, parse_action


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
    provenance, reasons, actions = [], Counter(), Counter()
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
            full = messages + [{"role": "assistant", "content": decision["output"]}]
            tokens = tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False)
            prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if tokens[:len(prefix)] != prefix or len(tokens) <= len(prefix):
                raise ValueError("chat template does not support exact assistant-only loss masking")
            if len(tokens) > max_tokens:
                reasons["overlength_turn"] += 1
                continue
            examples[split].append({"messages": full})
            provenance.append({"split": split, "question_id": qid, "turn": decision["turn"],
                               "tokens": len(tokens), "loss_start": len(prefix),
                               "action": action["action"]})
            lengths.append(len(tokens))
            supervised.append(len(tokens) - len(prefix))
            actions[action["action"]] += 1
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
             "max_sequence_tokens": max(lengths), "supervised_tokens": sum(supervised),
             "mask_prompt": True, "tool_observations_are_targets": False,
             "quality_filter": "status=ok and train GoldHit@10=1; this does not verify semantic answer correctness",
             "file_sha256": {name: sha256(out_dir / name) for name in ("train.jsonl", "valid.jsonl")},
             "source_sha256": {name: sha256(data_dir / name) for name in
                                ("train.jsonl", "train_labels.jsonl")}}
    (out_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats
