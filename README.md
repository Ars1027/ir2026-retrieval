# IR 2026 检索大作业

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

## 本组基础实现（BM25 / dense / hybrid）

三个固定流水线共用一套证据、答案与结果组装代码；每题仍是 `retrieval` 事件，不使用模型自主工具交互。

```text
retrieval/contracts.py           Chunk / Hit / RankedDoc 字段与校验
retrieval/corpus_adapter.py      只读语料访问
retrieval/chunking.py            保真切块（原文[start:end]）与稳定 chunk_id
retrieval/bm25.py                倒排索引、BM25Retriever
retrieval/document_rank.py       片段到文档的稳定聚合
retrieval/embedding.py           编码器（hf / hashing）
retrieval/dense.py               FAISS 精确内积索引、DenseRetriever
retrieval/fusion.py              文档级 RRF
pipeline/settings.py             IR2026_METHOD_CONFIG 配置读取
pipeline/evidence.py             逐字原文引用
pipeline/answer.py               extractive / model 两种答案策略
pipeline/model_client.py         OpenAI 兼容客户端（密钥读环境变量）
pipeline/result_builder.py       共用 assemble_result
pipeline/hybrid.py               固定 hybrid 流水线
agent/{bm25,dense,hybrid}_solution.py, agent/solution.py  课程适配器与默认入口
```

构建（产物写入 `artifacts/`，不提交）：

```bash
python scripts/build_chunks.py
python scripts/build_bm25.py
python scripts/build_dense.py
```

运行（`IR2026_METHOD_CONFIG` 由脚本自动导出，输出目录必须为空）：

```bash
python scripts/run_baseline.py --method bm25   --split dev --output output/member_b/dev_bm25   --check --evaluate
python scripts/run_baseline.py --method dense  --split dev --output output/member_b/dev_dense  --check --evaluate
python scripts/run_baseline.py --method hybrid --split dev --output output/member_b/dev_hybrid --check --evaluate
python scripts/run_baseline.py --method final  --split test --output output/member_b/test_base_final --check
bash run.sh --method hybrid --split dev --output output/member_b/dev_hybrid      # 或 .\run.ps1 ...
```

汇总与测试：

```bash
python scripts/summarize_runs.py --run-dir output/member_b/dev_hybrid --split dev
python -m unittest discover -s tests -v            # 课程原始测试
python -m unittest discover -s tests/member_a -v   # 成员A模块测试
python -m unittest discover -s tests/member_b -v   # 成员B模块测试
```

接口约定：`start/end` 为 `Corpus.text(doc_id)` 的 Python 字符位置，`doc_id` 保持课程原始字符串，`rank` 从 1 开始，`score` 仅在同一召回器内可比。`agent/solution.py` 是冻结的默认入口（当前为 BM25）；`configs/hybrid.json` 与 `configs/dense.json` 保留给需要真实语义编码器的实验，索引元数据会记录实际使用的编码器。
