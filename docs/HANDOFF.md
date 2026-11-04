# 附加题交接

## 基础阶段交付

默认入口 `agent/solution.py` 读取 `configs/final.json`，执行固定检索流程：BM25 召回 → 片段聚合 → BGE-M3 向量召回 → 文档级 RRF → 原文证据 → 抽取式答案。`agent/bm25_solution.py` 保留独立 BM25 对照。

| 职责 | 代码与配置 |
| --- | --- |
| 原文、切块、ID 与字符跨度 | `retrieval/corpus_adapter.py`、`chunking.py`、`contracts.py` |
| BM25 与文档聚合 | `retrieval/bm25.py`、`document_rank.py`、`configs/bm25.json` |
| API 编码、FAISS、RRF | `vector_backend/ada.py`、`scripts/build_vector.py`、`configs/vector.bge.json` |
| 向量插件契约与流水线 | `retrieval/interfaces.py`、`pipeline/vector_stage.py` |
| 引用、答案、公共结果组装 | `pipeline/evidence.py`、`answer.py`、`model_client.py`、`result_builder.py` |
| 配置与运行 | `pipeline/settings.py`、`scripts/run_baseline.py`、`run.ps1`、`run.sh` |
| 检查、评测、汇总 | `check_results.py`、`evaluate.py`、`scripts/summarize_runs.py` |

`vector_backend/ada.py` 是沿用的文件名，实际模型由配置决定。当前使用 BGE-M3；ada-002 的部分构建记录作为历史记录保留，使用独立索引目录。

## 附加题一：检索方法改进

当前向量检索与融合已是基础交付的一部分。附加题从当前 BM25 和混合检索两套完整 dev50 结果出发，选择新的检索、重排、查询改写或多轮 Agent 方法，安排对照与消融。

- 新增方法配置，从 `configs/vector.bge.json` 复制参数，设置新的方法名、索引目录和输出目录。
- 向量后端实现 `retrieval.interfaces.VectorRetriever`；文档融合实现 `DocumentFusion`，通过 `module:factory` 配置加载。详细字段见[接口指南](../VECTOR_PLUGIN_GUIDE.md)。
- 固定流程使用 `retrieval` 日志；模型自主选择工具时使用 `agent` 和 `model_tool` 日志。当前统一入口固定为 retrieval，Agent 方法可通过课程 `run.py --method-type agent --model-label <实际模型>` 运行新的适配器。
- 先 train 冒烟，再完整 dev50，报告 Hit@1/5/10、EvidenceRecall@10、AnswerEM、耗时、token 与失败分布。

当前可改进的问题：混合检索 Hit@10 提升到 0.44，但 Hit@1 降至 0.16；抽取式答案 AnswerEM 为 0。模型答案客户端已实现，示例配置在 `configs/model.example.json`，生成式答案尚无本项目正式实验结果。

## 附加题二：小模型后训练

可复用原文访问、检索候选、证据组装、结果协议和评测程序。下一阶段需要实现训练数据构建、训练脚本、训练前后评测及权重或适配器交付；教师轨迹采集与 SFT/LoRA 尚未实现。已有 retrieval trace 是固定检索流程日志，训练样本的字段和监督目标需单独设计。

训练问题和标签用于训练/筛选，dev 用于选择参数；test 在方法冻结后运行。原文、QA、引用和逐题轨迹按 [DATA_USE.md](../DATA_USE.md) 在课程内部流转。

## 共同接口与验收

- `solve(question, tools, config)` 接收问题，返回答案、排序文档、证据、状态和 usage。
- `doc_id` 使用课程原始字符串；`start/end` 是 `Corpus.text(doc_id)` 的 Python 字符位置。引用直接取连续原文切片。
- 每题 600 秒，最多 10 篇文档、5 段引用，每段 20–1200 字符。模型工具最多 32 次，每次观测最多 12,000 字符。
- token 来自实际 API usage，未知值使用 `null`。失败题保留在完整评测中。
- 新方法先完成完整 dev、官方检查和独立复跑，再保存冻结配置与最终 test 输出。基础配置 `configs/final.json` 用作固定对照，新方法使用单独配置。

课程的两道选做各最高 10 分：检索改进包括方案实现、对照消融与分析；后训练包括数据、实际训练和训练前后评测。具体评分见 [SPEC.md](../SPEC.md)。

## 接手时需要的文件

Git 包含代码、配置、依赖、合成测试和聚合实验记录。课程数据从课程渠道获取；索引可按 README 重建；已有逐题结果由当前工作区交接。具体路径和远程核对结果见[文件清单](REPOSITORY_STATUS.md)。
