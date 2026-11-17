"""Local coursework package, including private cases and the inspectable adapter."""
from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

from ir2026 import keyed, load_questions, read_jsonl
from pipeline.settings import resolve_path
from posttraining.data import sha256
from posttraining.report import compare_runs, describe_run


def make_delivery(root, before, after, test, teacher, output):
    root, before, after, test, teacher, output = map(Path, (root, before, after, test, teacher, output))
    if output.exists():
        raise ValueError("delivery archive already exists; use a new name")
    comparison = compare_runs(root, before, after)
    test_summary, _ = describe_run(root, test, split="test")
    describe_run(root, teacher, split="train")
    after_config = json.loads((after / "method_config.json").read_text())
    adapter = resolve_path(after_config["model"]["adapter_path"])
    training = json.loads((adapter / "training_summary.json").read_text())
    if training["adapter_sha256"] != sha256(adapter / "adapters.safetensors"):
        raise ValueError("trained adapter changed before packaging")
    info = json.loads((test / "run_info.json").read_text())
    if training["adapter_sha256"][:16] not in info["model"]:
        raise ValueError("test run used a different adapter")
    labels = keyed(read_jsonl(root / "datas/dev_labels.jsonl"), "dev labels")
    questions = keyed(load_questions(root / "datas/dev.jsonl"), "dev")
    old = keyed(read_jsonl(before / "results.jsonl"), "before")
    new = keyed(read_jsonl(after / "results.jsonl"), "after")

    def hit(rows, qid):
        return bool(set(rows[qid]["doc_ids"][:10]) & set(labels[qid]["gold_doc_ids"]))

    selected = []
    for kind, predicate in (
        ("训练后新增命中", lambda q: not hit(old, q) and hit(new, q)),
        ("训练前后均命中", lambda q: hit(old, q) and hit(new, q)),
        ("训练后丢失命中", lambda q: hit(old, q) and not hit(new, q)),
        ("训练后仍未命中", lambda q: not hit(new, q)),
    ):
        choice = next((q for q in questions if q not in {i[1] for i in selected} and predicate(q)), None)
        if choice:
            selected.append((kind, choice))
    for qid in questions:
        if len(selected) >= 4:
            break
        if qid not in {i[1] for i in selected}:
            selected.append(("补充对照", qid))
    lines = ["# 附加题 2：基于教师轨迹的小模型后训练", "",
             "本报告是小组报告的附加题 2 章节。题目、答案、引用和逐题轨迹仅在课程内部交付。", "",
             "实现与复现说明见 source/docs/POSTTRAINING.md。基础模型、课程语料和标签未重复打包。", "",
             "## 数据与实际训练", "",
             f"教师为本地 Qwen3-4B-Instruct-2507（4-bit），学生为本地 Qwen2.5-1.5B-Instruct（4-bit）。只从 train100 采集教师轨迹。",
             f"质量筛选后，训练/内部验证题数为 {training['data']['accepted_questions']}，逐步样本数为 {training['data']['examples']}。",
             f"监督来源统计：{training['data']['target_sources']}。最终答案只在当前输入的 gold 原文可逐字核验时保留；必要时明确使用训练标签校正，原始教师输出保留在决策文件。",
             f"实际完成 {training['selected_checkpoint']['iteration']} 次更新，耗时 {training['wall_seconds']:.1f} 秒。LoRA 参数与软件版本见 training_summary.json。",
             f"适配器大小 {training['adapter_bytes']} bytes；SHA256 `{training['adapter_sha256']}`。", "",
             "以历史观测预测下一步动作和答案；只对最后一条 assistant 消息计算损失，工具返回和历史动作全部屏蔽。按原始训练题划分内部验证，不使用课程 dev/test 题训练或筛选教师轨迹。", "",
             "## 完整 dev50 对照", "",
             "| 指标 | 训练前 | 训练后 |", "| --- | ---: | ---: |"]
    for key in ("gold_hit_at_1", "gold_hit_at_5", "gold_hit_at_10", "evidence_recall_at_10",
                "answer_em", "completion_rate", "mean_tool_calls", "mean_elapsed_seconds"):
        lines.append(f"| {key} | {comparison['before']['metrics'][key]:.6f} | {comparison['after']['metrics'][key]:.6f} |")
    for key in ("model_calls", "invalid_actions", "fallback_rankings"):
        lines.append(f"| {key} | {comparison['before'][key]} | {comparison['after'][key]} |")
    lines += ["", "两次均完整覆盖 50 题并通过官方 submission 检查。提示词、基础权重、索引、工具、greedy 解码与预算一致，仅 LoRA 不同。模型每题重新加载，计入单题耗时；失败保留在分母和日志中。", "",
              f"训练前 token：{comparison['before']['tokens']}；训练后 token：{comparison['after']['tokens']}。本地模型 API 费用为 0，未计电费。",
              f"逐题 GoldHit@10 转移统计：{comparison['paired_gold_hit_at_10']}。", "",
              "原基础混合检索 dev Hit@10=0.44 是不同检索流程的参考结果；本节的训练前后因果对照使用同一 BM25 工具 Agent，不能将两套流程的差异归因于后训练。", "",
              "## 四个逐题案例", "",
              "以下覆盖实际存在的命中和失败类型。命中文档不等于答案正确；AnswerEM 也不等同语义正确率。"]
    for index, (kind, qid) in enumerate(selected, 1):
        result = new[qid]
        decisions = read_jsonl(after / "decisions" / f"{qid}.jsonl")
        lines += ["", f"### 案例 {index}：{kind}（{qid}）", "",
                  "问题：" + questions[qid]["question"], "", "验证标签答案：" + labels[qid]["answer"], "",
                  f"训练前/后 Hit@10：{int(hit(old, qid))}/{int(hit(new, qid))}；训练后状态：`{result['status']}`。",
                  "", "训练前短答案：" + (old[qid]["answer"] or "（空）"),
                  "", "训练后短答案：" + (result["answer"] or "（空）"),
                  "", "gold 文档：" + ", ".join(labels[qid]["gold_doc_ids"]),
                  "", "训练后排序：" + ", ".join(result["doc_ids"]), "", "实际模型动作：", ""]
        for decision in decisions:
            lines.append(f"- 回合 {decision['turn'] + 1}：`{decision.get('output') or '（生成失败）'}`")
        lines += ["", "实际返回给模型、并进入最终输出的原文引用：", ""]
        for evidence in result["evidence"][:2]:
            lines += [f"> 文档 {evidence['doc_id']}", ">", *["> " + s for s in evidence["text"].splitlines()], ""]
        if result["status"] == "budget_exceeded":
            lines.append("预算分析：未在动作/回合预算内完成终止，保留实际检索排名作为部分结果；失败仍在评测分母中。")
        if not hit(new, qid):
            lines.append("检索分析：最终前 10 篇未覆盖标注答案文档，这是可确认的召回/排序失败。")
        else:
            lines.append("检索分析：最终前 10 篇覆盖了标注答案文档；还需结合上面的阅读动作和引用核查是否读到关键证据。")
    lines += ["", "## 冻结 test50 与限制", "",
              f"最终 test 完整覆盖 {test_summary['queries']} 题，检查错误数为 {test_summary['validation_errors']}，状态分布为 {test_summary['status_counts']}。",
              "测试标签由教师持有，本地不计算测试命中率或正确率；测试前冻结适配器 SHA256，不据测试结果调参。",
              "本实验只有 100 道原始训练题、一个训练种子和一套小模型；训练步骤多于独立监督题数，存在过拟合风险。GoldHit 过滤仅验证文档命中，最终答案监督另做可见原文校验和训练标签校正。原文包含答案也不自动证明模型验证了所有问题条件；引用检查仅证明连续匹配。", ""]
    report = after.parent / "BONUS2_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root).decode().split("\0")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_files:
            path = root / name
            if name and path.is_file() and not path.is_symlink():
                archive.write(path, "source/" + name)
        for name, directory in (("before_dev", before), ("after_dev", after), ("test_final", test), ("teacher_train", teacher)):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, f"runs/{name}/" + str(path.relative_to(directory)))
        for name in ("adapters.safetensors", "adapter_config.json", "training.log", "training_summary.json"):
            archive.write(adapter / name, "adapter/" + name)
        data_dir = resolve_path(training["training"]["data"])
        for path in data_dir.iterdir():
            if path.is_file():
                archive.write(path, "training_data/" + path.name)
        archive.write(report, "BONUS2_REPORT.md")
    return {"archive_bytes": output.stat().st_size, "archive_sha256": sha256(output),
            "report": str(report), "archive": str(output)}
