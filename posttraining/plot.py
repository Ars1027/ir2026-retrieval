"""Standalone scientific plots from aggregate records, without benchmark text."""
from __future__ import annotations

import json


def plot_records(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = root / "records/posttraining"
    train = json.loads((directory / "training_summary.json").read_text())
    comparison = json.loads((directory / "dev_comparison.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "ir2026-bonus2"})
    fig, (loss_ax, metrics_ax) = plt.subplots(1, 2, figsize=(12.5, 4.1), layout="constrained")
    curve = train["training_loss"]
    loss_ax.plot([r["iteration"] for r in curve], [r["loss"] for r in curve],
                 color="#246B97", label="Training (mean over reporting interval)")
    curve = train["validation_loss"]
    valid_questions = train["data"]["accepted_questions"]["valid"]
    valid_states = train["data"]["examples"]["valid"]
    loss_ax.plot([r["iteration"] - 1 for r in curve], [r["loss"] for r in curve],
                 color="#C45D28", marker="o", markersize=4,
                 label=f"Held-out train ({valid_questions} question, {valid_states} states)")
    loss_ax.set(xlabel="Optimizer updates", ylabel="Assistant-target cross entropy",
                title="(a) Actual QLoRA training")
    loss_ax.grid(axis="y", alpha=0.2)
    loss_ax.legend(fontsize=8, frameon=False)
    keys = ["gold_hit_at_1", "gold_hit_at_5", "gold_hit_at_10", "evidence_recall_at_10", "answer_em"]
    labels = ["Hit@1", "Hit@5", "Hit@10", "Evidence\nRecall@10", "AnswerEM"]
    x = np.arange(len(keys))
    for shift, name, color in ((-0.2, "before", "#246B97"), (0.2, "after", "#C45D28")):
        values = [comparison[name]["metrics"][key] for key in keys]
        bars = metrics_ax.bar(x + shift, values, width=0.36, color=color, label=name.capitalize())
        metrics_ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8)
    metrics_ax.set(xticks=x, xticklabels=labels, ylabel="Full dev50 score", ylim=(0, 1.08),
                   title="(b) Same student, tools and decoding")
    metrics_ax.grid(axis="y", alpha=0.2)
    metrics_ax.set_axisbelow(True)
    metrics_ax.legend(frameon=False)
    fig.savefig(directory / "training_and_dev.png", dpi=180)
    svg_path = directory / "training_and_dev.svg"
    fig.savefig(svg_path, metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)
