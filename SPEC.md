# 数据接口与评测标准

## 任务与版本

输入为自然语言问题和固定本地语料，输出为最多 10 篇按相关性排序的文档、基于原文的简短答案和引用证据。方法、实现框架不限，不要求 harness、大模型或基础任务训练。

数据版本：`IR2026-BCP-200-v1`。训练 100 题、验证 50 题、测试 50 题，共用 10,000 篇文档。运行协议：`ir2026.retrieval.v2`。协议更新没有改变问题、语料、标注或核心指标公式。

## 预算与统计口径

| 约束 | 适用范围 | 每题上限 |
| --- | --- | ---: |
| 总运行时间 | 所有方法 | 600 秒 |
| 最终排序文档 | 所有方法 | 10 篇 |
| 最终引用 | 所有方法 | 5 段，每段 20–1,200 字符 |
| 模型发起的工具调用 | Agent 工具交互 | 32 次 |
| 单次返回给模型的工具观测 | Agent 工具交互 | 12,000 字符 |

普通检索可以使用索引、向量模型、重排模型或固定流水线，其内部步骤不按模型工具调用计数。独立的模型工具调用分别计数；一次模型发起的 Bash 调用内部的管道算一次工具调用。

单题时间覆盖实际查询处理，包括召回、排序、读取、可选答案生成及查询内的模型调用。可提前完成索引构建、语料编码和模型加载，其耗时与资源另行报告；不能隐藏发生在查询过程中的计算。项目示例运行器将进程启动及模块加载纳入超时等待，重型系统也可自行批量运行并据实计时。

配置见 `configs/default.json`，不得放宽正式评测预算。检查允许 5 秒进程收尾开销，不是额外检索时间。失败和超时保留记录与已完成日志，无法取得完整统计时标记 `tool_calls_complete=false`。

字符按 Python Unicode 字符计算。结构化观测长度为：

```python
len(json.dumps(output, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
```

无模型的方法将 token 记为 0。使用模型但无法取得 token 时填 `null` 并提供非空 `token_notes`；不得用 0 表示未知。报告说明模型版本及输入、输出、缓存、推理 token 的统计口径，不能仅按调用次数比较费用。

## 问题与标签

`datas/{train,dev,test}.jsonl` 每行包含 `id`、`question`。ID 必须保留为字符串，不能重新编号。训练及验证标签包含 `id`、`answer`、`gold_doc_ids`、`evidence_doc_ids`。

- `gold_doc_ids`：标注为包含最终答案的文档。
- `evidence_doc_ids`：支持回答的证据文档，可能包括中间线索。
- 标注不穷尽所有相关文档，未标注不等于错误证据。

运行时只能使用问题与语料。训练标签可用于离线训练、数据筛选和评测；验证标签用于评测与配置选择。测试标签、上游对应答案和现成轨迹不得进入检索环境。

## 结果目录与运行元数据

```text
test_final/
  run_info.json
  results.jsonl
  traces/
    查询ID.jsonl
```

无模型方法的 `run_info.json` 示例：

```json
{
  "release": "IR2026-BCP-200-v1",
  "protocol_version": "ir2026.retrieval.v2",
  "mode": "coursework",
  "method": "bm25-v1",
  "method_type": "retrieval",
  "agent_module": "agent.solution",
  "model": null,
  "split": "test",
  "limits": {
    "max_tool_calls": 32,
    "max_tool_output_chars": 12000,
    "max_seconds": 600,
    "max_doc_ids": 10,
    "max_evidence_quotes": 5,
    "min_quote_chars": 20,
    "max_quote_chars": 1200
  }
}
```

| 字段 | 要求 |
| --- | --- |
| `release`、`protocol_version` | 与项目版本一致 |
| `mode` | 正式提交为 `coursework`，演示为 `demo` |
| `method` | 本组实际方法名称/版本，应与代码和报告对应 |
| `method_type` | `retrieval` 为固定检索流水线；`agent` 为模型自主选择工具动作 |
| `model` | 未使用模型为 `null`；使用模型时记录实际标识，多个模型可用分号分隔并在报告中展开；Agent 必须填写 |
| `agent_module` | 项目运行器记录实际模块；外部程序可省略，另在报告说明入口 |
| `split` | `train`、`dev` 或 `test`，与检查命令一致 |
| `limits` | 完整复制固定配置中的 `limits`，模型专用限制仅作用于相应事件 |

`retrieval` 并不等于“禁止模型”：向量编码、重排、固定查询改写或答案生成可以使用模型，须据实填 `model` 与成本。只要模型自主发起工具交互，应使用 `agent` 并记录真实模型工具事件。

原样示例不能作为作业提交。`demo`、`examples.*` 模块和旧的 `offline-keyword-demo` 标识均会被学生端和教师端拒绝。仅更改元数据不代表完成作业，系统实现与报告仍需人工检查。

## 逐题结果

每题一条 JSONL 记录，顺序不限，但不得漏题、重复或增加未知 ID。以下是字段结构示例，并非真实可提交答案：

```json
{
  "id": "原始查询ID",
  "answer": "简短最终答案",
  "doc_ids": ["原始文档ID"],
  "evidence": [{"doc_id": "原始文档ID", "text": "原文中的连续片段"}],
  "status": "ok",
  "usage": {
    "tool_calls": 0,
    "trace_events": 2,
    "elapsed_seconds": 1.5,
    "model_input_tokens": 0,
    "model_output_tokens": 0,
    "tool_calls_complete": true,
    "token_notes": "No model used."
  }
}
```

- `status` 为 `ok`、`no_answer`、`timeout`、`error` 或 `budget_exceeded`。`ok` 必须有非空答案；检索到了文档但无法确定答案时用 `no_answer`。失败也保留在指标分母中。
- `doc_ids` 最多 10 个，不重复，必须存在于语料并出现在本题实际运行日志中。无需补足 10 个。
- 只要返回文档，就提供至少一段对应引用。每段引用的文档必须在最终列表内，文本逐字匹配该文档的连续原文，不能拼接或改写。
- 检查通过证明文档和引用格式有效，不证明引用支持答案；语义支持度需通过案例分析或教师复核。
- `tool_calls` 只统计 `model_tool` 事件；普通检索为 0。新协议必须填写 `trace_events`，记录全部日志事件数并与日志一致。

## 逐题运行日志

所有方法都提供 `traces/查询ID.jsonl`。日志记录实际发生的步骤，不能为通过检查而伪造。无任何已完成步骤的失败题提供空文件。

| `event_type` | 含义 | 计入 32 次限制 | 12,000 字符限制 |
| --- | --- | --- | --- |
| `retrieval` | 固定流水线内部的召回、排序、读取等步骤 | 否 | 否 |
| `model_tool` | 模型发起的工具调用及其返回观测 | 是 | 是 |

Agent 内部可以另存 `retrieval` 步骤，但向模型返回的工具结果必须有对应 `model_tool` 事件，不能用内部日志替代。项目运行器在 `agent` 模式下，将通过 `tools.search/read` 的每次调用记为模型工具交互；普通方法使用 `retrieval` 模式即可。

字段结构如下；`output_chars` 必须按实际输出计算，不能照填示例数字：

```json
{
  "id": "原始查询ID",
  "seq": 1,
  "event_type": "retrieval",
  "tool": "bm25_search",
  "arguments": {"query": "问题文本", "top_k": 10},
  "output": {"hits": [{"doc_id": "原始文档ID", "score": 3.2}]},
  "doc_ids": ["原始文档ID"],
  "error": null,
  "elapsed_ms": 12.0,
  "output_chars": 0
}
```

`seq` 从 1 连续编号。`doc_ids` 记录该步骤实际涉及并返回的文档。日志可以只保留必要的候选 ID、分数和摘要，不必重复写入整篇语料；最终文档仍须能追溯到日志。模型工具事件保存模型实际收到的截断后观测。完整原始输出可另存并标明对应关系；不得要求导出不可见的内部思维链。

通过项目接口访问自己的索引时，普通检索可使用 `tools.record_retrieval(stage, arguments, output, doc_ids, elapsed_ms)`；它记录已执行的步骤，不替你执行检索。Agent 或外部框架也可直接导出同格式日志。检索方法、实际模型交互与日志类型应一致，教师会结合代码和报告核对。

## 指标与评分

**GoldHit@k（简称 Hit@k）**：每题前 k 篇文档与 `gold_doc_ids` 有交集记 1，否则记 0；再对全部问题取平均。报告 k=1、5、10，主指标是 GoldHit@10。

**EvidenceRecall@10**：每题前 10 篇覆盖的标注证据数除以该题标注证据总数，再取平均。它衡量给定标注下的召回，不代表所有相关文档的完整召回率。

答案严格匹配作为辅助指标保留，不计入自动效果分。`completion_rate` 是 `status=ok` 的比例，不表示答对。运行统计分别报告平均模型工具调用 `mean_tool_calls`、日志事件 `mean_trace_events`、单题耗时，以及统计不完整的题数。

基础作业 30 分：系统实现 12 分、实验效果 8 分、报告 10 分。效果分中的 6 分为 `6 × GoldHit@10`，另 2 分检查全量提交、格式、引用和日志一致性。系统实现和报告由教师检查，不以是否使用大模型或 harness 作为得分前提。

- 检索方法改进选做：最高 10 分，方案与实现 4 分、对照与消融 4 分、分析 2 分。
- 小模型后训练选做：最高 10 分，数据 2 分、实际训练 4 分、训练前后评测及分析 4 分。
- 加分可补充平时和期末成绩，不做不扣分。没有提升也应如实分析。

验证集报告全部 50 题；测试集提交全部 50 题，由教师统一评测。禁止删除失败题、硬编码测试答案、用测试问题训练或根据测试反馈调参。

## 兼容性

旧版不含 `protocol_version`、`method_type` 的历史运行可继续检查，其无类型日志全部按旧版模型工具调用计数，预算不放宽。新格式必须显式填写协议和事件类型。无模型正式方法应使用本版格式，不要伪造模型标识或仅修改旧日志的计数字段。

学生检查程序、验证集评测和教师评分共用 `ir2026.py` 的元数据与结果校验。教师必须使用配套的本版学生项目和教师包，不能混用旧评测器。
