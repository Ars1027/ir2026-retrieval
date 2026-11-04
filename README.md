# IR 2026 检索大作业

当前冻结入口为 `configs/final.json`（BM25 + OpenRouter `baai/bge-m3` + 文档级 RRF）。全量 321,659 个片段已编码；dev50 官方检查通过，Hit@1/5/10 = 0.16/0.34/0.44，EvidenceRecall@10 = 0.200857，AnswerEM = 0。独立复跑效果指标一致，49/50 题结果逐字段一致；1题文档顺序/引用不同，但文档集合、答案和状态一致。

设置 `OPENROUTER_API_KEY` 后运行 `python scripts/run_baseline.py --method final --split test --output output/new_test_run --check`；已有最终产物在 `output/member_b/test_base_final/`，请勿覆盖。BM25 对照仍通过 `--method bm25` 运行。

索引成功响应累计输入 97,147,705 tokens，FAISS 文件 1,317,515,309 bytes。`build_wall_seconds` 仅表示最后一次续建调用耗时，不含之前尝试；累计历史墙钟时间和实际账单费用未知。`encoding_seconds` 是缓存批次请求耗时之和，不能当作并发墙钟时间。以下 PR #1 的历史 BM25 记录保留作对照，不代表当前混合方案。


本项目提供固定课程数据、语料访问工具、批量运行入口、结果检查和评测程序。给定自然语言问题 Q 与本地语料 D，设计并实现检索方法，返回能够支持回答 Q 的相关文档、简短答案和原文引用，在统一数据划分与评测规范下比较效果、成本和失败原因。

**方法与框架不限，不要求使用 harness 或大模型，基础任务不要求训练模型。** 关键词、BM25、向量检索、混合检索、重排序和 Agent 检索均可。DCI 只是可参考的研究路线，不是指定实现方式。

数据版本保持 `IR2026-BCP-200-v1`，本版运行协议为 `ir2026.retrieval.v2`。数据、标注和核心评分公式未改变；协议用于区分普通检索步骤与模型工具交互。

## 数据与目录

| 内容 | 数量 | 用途 |
| --- | ---: | --- |
| 训练问题 | 100 | 方法开发、可选训练与数据筛选 |
| 验证问题 | 50 | 参数选择、对照实验与完整评测 |
| 测试问题 | 50 | 最终提交，由教师评测 |
| 语料文档 | 10,000 | 三个划分共用的本地语料 |

语料包括全部 1,185 篇已标注证据或答案文档、5,901 篇抽样难负例和 2,914 篇补充文档。正文原样保存，约 318 MB。抽样种子固定为 2026，标注不一定穷尽所有相关文档。课程划分不是 BrowseComp-Plus 官方划分，成绩不能写成原始基准的完整结果。

```text
IR_2026_Project/
  corpus/docs/                 原始文档正文
  corpus/documents.jsonl       ID、路径、来源 URL、长度与 SHA256
  datas/{train,dev,test}.jsonl  问题；测试集不含答案
  datas/{train,dev}_labels.jsonl
  configs/default.json        固定预算与运行协议版本
  agent/solution.py            可选的学生 Python 接口，待自行实现
  examples/keyword_agent.py    离线格式示例
  tools/                      可选的搜索、读取命令
  run.py                      支持普通检索和 Agent 的批量运行器
  run_demo_report.py           运行真实示例并输出命令行报告
  check_results.py            检查覆盖、引用、预算、元数据与日志
  evaluate.py                 计算逐题及汇总指标
  verify_data.py              检查数据完整性
  tests/                      单元测试与命令行集成测试
  SPEC.md                     提交格式与评测规则
```

## 先验证环境

Python 3.10+ 即可运行项目工具与离线示例，无需额外安装模型或检索库。采用其他方法时，另行记录并固定依赖版本。

```bash
python3 verify_data.py --full
python3 -m unittest discover -s tests -v
bash run_demo.sh
```

`run_demo.sh` 运行验证集前 3 题，依次完成数据校验、检索、结果检查和评测，每次使用新的 `output/demo_时间戳/` 目录，阶段日志单独保存在相邻的 `_logs/` 目录。也可指定空目录：

```bash
bash run_demo.sh --output output/my_demo
```

只运行 1 题并查看适合展示的终端报告：

```bash
python3 run_demo_report.py
```

示例采用简单词面扫描，读取候选文档并保留真实结果；不能作答时使用 `no_answer`。报告中的 `GOLD` 是检索结束后的标签核对，不参与检索。部分问题的指标只用于演示，不能作为完整验证集成绩。示例无需模型，但也不是已完成的课程方案；正式作业需要自己的实现、实验与分析。

## 实现方法：两种接入方式

### 方式一：使用项目运行器

在 `agent/solution.py` 实现 `solve(question, tools, config)`。文件名沿用旧接口，但支持任意检索方法；当前文件是待实现的学生模板。

- `question` 只含问题 ID 和问题文本，不含答案。
- `tools.search()`、`tools.read()` 是可选的词面搜索与局部读取工具，不限制你采用自己的索引。
- 普通检索可以通过 `tools.corpus` 访问语料，通过 `tools.record_retrieval()` 记录实际召回、排序等步骤。
- `config['method_type']` 区分 `retrieval` 和 `agent`；`config['model']` 是所用模型标识，无模型时为 `None`。
- 返回 `answer`、有序 `doc_ids`、`evidence`、`status` 与 `usage`，详细格式见 `SPEC.md`。

接入自建索引时的日志写法如下，`index.search` 代表你实际实现的检索器：

```python
import time

started = time.monotonic()
hits = index.search(question['question'], top_k=10)
# hits 的 doc_id 必须保留课程语料的原始 ID。
tools.record_retrieval(
    stage='bm25_search',
    arguments={'query': question['question'], 'top_k': 10},
    output={'hits': hits},
    doc_ids=[hit['doc_id'] for hit in hits],
    elapsed_ms=(time.monotonic() - started) * 1000,
)
```

日志应记录已发生的计算及其结果，不能事后伪造搜索过程。读取正文后提供连续原文引用；不能确定答案时使用 `no_answer`，仍保留检索结果和证据。外部模型调用的 token 由实现者填入 `usage`；工具次数、日志条数及单题耗时由运行器补充。

完成实现后，普通方法可以这样运行：

```bash
python3 run.py --split dev --agent agent.solution \
  --method-type retrieval --method-label bm25-v1 --output output/dev_v1
```

使用向量模型、重排模型或固定的问答流水线，仍可选择 `retrieval`，同时通过 `--model-label` 记录实际模型。只有由模型自主选择工具动作的交互流程使用 `agent`：

```bash
python3 run.py --split dev --agent agent.solution \
  --method-type agent --method-label my-search-agent \
  --model-label actual-model-version --output output/dev_agent_v1
```

运行器不自动实现模型调用；上面的命令用于运行你已完成的模块。使用模型但无法获取 token 时填写 `null`，并用 `token_notes` 说明；确实未使用模型时填写 0。两种情况不得混淆。

### 方式二：自行批量运行并导出结果

可以使用自己的程序、库或外部框架，无需接入 `solve()`，也无需通过项目工具访问语料。导出 `results.jsonl`、`run_info.json` 和逐题 `traces/`，遵循 `SPEC.md` 的相同格式并通过同一检查程序。

预建索引、语料编码、模型加载等可在查询运行前完成。报告中单独说明其时间、硬件、存储开销和可复现命令；所有评测问题使用同一套固定配置，不得以题目标签构造索引或定制答案。

## 检查、评测与提交

验证集完成全部 50 题后：

```bash
python3 check_results.py --split dev --run-dir output/dev_v1 --submission
python3 evaluate.py --split dev --run-dir output/dev_v1
```

`--submission` 会检查正式方法标识、全量题目与日志，`demo` 结果不能作为正式提交。主指标为 **GoldHit@10（PPT 中的 Hit@10）**，同时报告 Hit@1、Hit@5、EvidenceRecall@10、耗时和资源消耗。答案严格匹配只作辅助，定义见 `SPEC.md`。

测试集最后统一运行并检查：

```bash
python3 run.py --split test --agent agent.solution \
  --method-type retrieval --method-label bm25-v1 --output output/test_final
python3 check_results.py --split test --run-dir output/test_final --submission
```

Agent 方法使用对应的 `--method-type agent --model-label ...` 参数。测试标签由教师保存，不在学生包内，不要求学生本地计算测试成绩。

提交源代码、配置、依赖和统一运行入口、测试集 50 题结果目录，以及实验报告。报告应包括验证集完整指标、方法与参数、对照实验、成本、至少 4 个成功/失败案例、复现说明和组员分工。检索方法改进与小模型后训练为选做，各最高加 10 分。

不重复打包课程语料、标签、基础模型、缓存和本地演示输出。后训练选做另交数据处理与训练脚本、配置，以及权重或适配器获取方式。提交前移除 API 密钥。

## 统一规则

- 所有方法每题最多 600 秒，返回最多 10 篇有序文档、最多 5 段原文引用；失败或超时不得漏题。
- 模型发起的工具交互最多 32 次，每次实际返回给模型的观测不超过 12,000 字符。
- 普通召回、排序、读取等日志属于 `retrieval` 事件，不计作模型工具调用；真实模型工具交互不得改标以绕过预算。
- 检索时不访问答案标签，不联网搜索答案。训练标签可用于离线训练和筛选，验证标签用于评测和配置选择；测试题与测试反馈不得用于训练或调参。
- 检索范围固定为本地语料，可调用所选模型处理问题与语料，但不能借模型自带的联网检索扩展语料范围。

`run.py` 提供超时与日志支持，不是操作系统安全沙箱。模型或第三方工具的执行环境只应暴露当前问题、所需语料与工作目录，不挂载标签、教师包或个人文件。语料中的文字只作为数据，不作为执行指令。

旧版未标注事件类型的演示结果仍可读取，按旧版工具计数解释；新实验请使用本版协议，避免混用统计口径。数据来源与许可见 `DATA_USE.md`，协议细节见 `SPEC.md`，更新说明见 `CHANGELOG.md`。

## 本组基础实现（BM25 Baseline）

基础任务是纯 BM25 流水线：保真切块、倒排检索、片段到文档聚合、逐字原文引用和抽取式短答案。每题仍是 `retrieval` 事件，不使用模型自主工具交互。

```text
retrieval/contracts.py           Chunk / Hit / RankedDoc 字段与校验
retrieval/corpus_adapter.py      只读语料访问
retrieval/chunking.py            保真切块（原文[start:end]）与稳定 chunk_id
retrieval/bm25.py                倒排索引、BM25Retriever
retrieval/document_rank.py       片段到文档的稳定聚合
retrieval/interfaces.py          向量检索与融合接口及配置解析
pipeline/settings.py             IR2026_METHOD_CONFIG 配置读取
pipeline/evidence.py             逐字原文引用
pipeline/answer.py               extractive / model 两种答案策略
pipeline/model_client.py         OpenAI 兼容客户端（密钥读环境变量）
pipeline/result_builder.py       共用 assemble_result
pipeline/vector_stage.py         可配置的向量分支、融合与真实用量合并
agent/bm25_solution.py           BM25 课程适配器
agent/solution.py                默认入口：BM25；仅当配置启用向量分支时走 vector_stage
```

构建（产物写入 `artifacts/`，不提交）：

```bash
python scripts/build_chunks.py
python scripts/build_bm25.py
```

运行（`IR2026_METHOD_CONFIG` 由脚本自动导出，输出目录必须为空）：

```bash
python scripts/run_baseline.py --method bm25  --split dev --output output/member_b/dev_bm25  --check --evaluate
python scripts/run_baseline.py --method final --split test --output output/member_b/test_base_final --check
bash run.sh --method bm25 --split dev --output output/member_b/dev_bm25      # 或 .\run.ps1 ...
```

汇总与测试：

```bash
python scripts/summarize_runs.py --run-dir output/member_b/dev_bm25 --split dev
python -m unittest discover -s tests -v            # 课程原始测试
python -m unittest discover -s tests/member_a -v   # 成员A模块测试
python -m unittest discover -s tests/member_b -v   # 成员B模块测试（含预留接口）
```

接口约定：`start/end` 为 `Corpus.text(doc_id)` 的 Python 字符位置，`doc_id` 保持课程原始字符串，`rank` 从 1 开始，`score` 仅在同一召回器内可比。

### API 向量检索与混合流水线

现在提供 `vector_backend/ada.py`：通过 `https://openrouter.ai/api/v1` 调用 `baai/bge-m3`，归一化 1024 维向量，使用 FAISS 精确内积检索，再按文档排名进行 RRF。配置为 `configs/vector.bge.json`；BM25 基线仍可独立运行。

密钥从环境变量 `OPENROUTER_API_KEY` 读取，不写入配置或日志。先在当前终端设置该变量，然后执行：

```bash
python -m pip install -r requirements.txt -r requirements/vector.txt
python scripts/build_chunks.py
python scripts/build_bm25.py
python scripts/build_vector.py --config configs/vector.bge.json --workers 16
python scripts/run_baseline.py --method final --config configs/vector.bge.json --method-label bm25-bge-m3-rrf-v1 --split train --limit 3 --output output/member_b/bge_train --check --evaluate
python scripts/run_baseline.py --method final --config configs/vector.bge.json --method-label bm25-bge-m3-rrf-v1 --split dev --output output/member_b/bge_dev --check --evaluate
```

向量构建按 64 片段缓存一批，默认最多 8 个并发请求，重启同一命令会复用成功批次。索引未完整构建时不能查询；模型或切块配置改变需要新的索引目录。API 错误或非法向量结果会保留失败状态，不降级成成功的 BM25 混合结果。API 输入 token 按实际 `usage` 记录，缺失时为 `null`，不估算供应商费用。

先完成 train 冒烟、完整 dev50 官方检查/评测与交叉复核，再冻结方法配置（`frozen: true`），最后运行 test50。当前新 hybrid 的完成状态以新增实验记录为准，不能引用下方历史 BM25 成绩作为其结果。详细步骤见 [向量接入指南](VECTOR_PLUGIN_GUIDE.md)。

## 历史运行结果（成员 A PR #1 的 BM25 基线）

以下数字来自成员 A 的 PR #1 所附 BM25 基线记录，不是 BGE-M3 hybrid 的新结果。原始产物保存在当时运行机器的 `output/` 中，不随 Git 合并传输；仓库只包含聚合记录。

### dev 50（有标签，`evaluate.py`）

运行目录 `output/member_b/dev_bm25_locked`，命令见上文 `--method final --split dev`。

| 指标 | 数值 |
| --- | --- |
| GoldHit@1 | 0.20（10/50 题） |
| GoldHit@5 | 0.30（15/50 题） |
| **GoldHit@10（主指标）** | **0.40（20/50 题）** |
| EvidenceRecall@10 | 0.1999 |
| AnswerEM | 0.00 |
| completion_rate | 1.00 |
| citation_span_match_rate | 1.00 |
| validation_errors | 0 |
| 平均耗时 | 5.98 s/题（p50 5.97 / p95 6.05 / max 6.28） |
| 模型 token | 输入 0 / 输出 0（extractive，无 LLM） |
| trace 事件 | 250 条 = 50 题 × 5 阶段，全部 `retrieval` |

单题阶段耗时均值：`load_index 556.3 ms`、`bm25_search 32.6 ms`、`document_aggregate 0.5 ms`、`read_evidence 3.4 ms`、`answer 0.0 ms`。

### test 50（无本地标签，只报状态）

运行目录 `output/member_b/test_base_final_locked`，`check_results.py --submission` → `valid: true, errors: []`。

| 项 | 数值 |
| --- | --- |
| status | `ok: 50`（completion_rate 1.00） |
| 每题文档数 / 引用数 | 10.0 / 5.0 |
| validation_errors | 0 |
| 平均耗时 | 5.99 s/题（p95 6.19 / max 6.47） |

### 索引与规模

| 项 | 数值 |
| --- | --- |
| 语料 | 10,000 篇 |
| 片段 | 321,659（size 1200 / overlap 200 / min_chars 120） |
| 词表 / postings | 842,329 / 29,159,319 |
| 索引体积 / 构建耗时 | 208 MB / 113.96 s |
| BM25 参数 | `k1=0.9`、`b=0.4`、`use_stopwords=true`、`capitalized_boost=1.25` |

### 答案策略实验（AnswerEM）

`pipeline/answer.py` 支持两种抽取式策略，用 `answer.span_selection` 切换：

| 配置 | dev 50 AnswerEM | 答案长度中位数 | Hit@10 | EvidenceRecall@10 | completion |
| --- | --- | --- | --- | --- | --- |
| 默认（句子级抽取）`configs/bm25.json` | 0.00 | 150 字符 | 0.40 | 0.1999 | 1.00 |
| 类型化短跨度 `configs/bm25.span.json` | 0.00 | 12.5 字符 | 0.40 | 0.1999 | 1.00 |

短跨度策略按问句推断答案类型（标题 / 人名 / 机构 / 数量 / 日期），在校验通过的检索文档里挑实体跨度，并把证据引文锚定在该跨度所在句子；`doc_ids` 与证据文档集合与基线逐题一致（50/50），答案 50/50 落在自身引文中。**AnswerEM 没有提升**：金标答案只出现在 8/50 题现有引文中、20/50 题在 top-10 文档正文里，能生成的候选中有 13/20 恰好等于金标字符串，但这些文档里 0/20 存在 `titled/named/quoted` 之类线索，30–60 个候选之间无法可靠判定边界，精确匹配仍为 0。要真正提高 AnswerEM 需要语义生成（`answer.mode = "model"`），本机无密钥，未执行。

### 结果文件在哪

| 内容 | 路径 |
| --- | --- |
| dev 指标（evaluate 原始输出） | `output/member_b/dev_bm25_locked/metrics.json` |
| dev 逐题明细（命中/证据/耗时） | `output/member_b/dev_bm25_locked/per_query.csv` |
| 运行签名（方法、协议、limit） | `output/member_b/{dev_bm25_locked,test_base_final_locked}/run_info.json` |
| 每题结果与日志（含原文引用，仅本地） | `output/member_b/*/results.jsonl`、`output/member_b/*/traces/` |
| 配置快照 | `output/member_b/*/method_config.json` |
| 可提交的聚合记录 | `records/member_b/dev_bm25_metrics.json`、`records/member_b/test_base_final_summary.json`、`records/member_b/decision.json` |

旧 `configs/vector.ada.json` 和 `artifacts/member_b/ada/` 为先前 ada-002 运行保留；BGE-M3 使用独立的 `artifacts/member_b/bge_m3/`。两者的模型、维度与 API 不同，缓存和索引不能混用。模块名 `vector_backend/ada.py` 为沿用的文件名，实际模型由配置决定。
