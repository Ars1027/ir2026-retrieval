# 项目交接

基础检索和附加题二已完成，实验分别见[基础结果](RESULTS.md)与[后训练结果](POSTTRAINING_RESULTS.md)。附加题一继续从当前 BM25 / 混合检索对照开展新方法与消融。

## 入口与模块

| 职责 | 代码与配置 |
| --- | --- |
| 冻结基础方法 | `agent/solution.py`、`configs/final.json` |
| BM25 对照 | `agent/bm25_solution.py`、`configs/bm25.json` |
| 原文、切块、ID 与字符跨度 | `retrieval/corpus_adapter.py`、`chunking.py`、`contracts.py` |
| BM25 与文档聚合 | `retrieval/bm25.py`、`document_rank.py` |
| API 编码、FAISS、RRF | `vector_backend/ada.py`、`scripts/build_vector.py`、`pipeline/vector_stage.py` |
| 引用、答案与结果组装 | `pipeline/evidence.py`、`answer.py`、`model_client.py`、`result_builder.py` |
| 基础运行与配置 | `pipeline/settings.py`、`scripts/run_baseline.py`、`run.ps1`、`run.sh` |
| 后训练 Agent | `agent/posttrained_solution.py`、`posttraining/`、`configs/posttraining/` |
| 后训练采集、训练与评测 | `scripts/posttrain.py`、`run_bonus2.sh` |
| 官方检查与评测 | `check_results.py`、`evaluate.py`、`scripts/summarize_runs.py` |

`vector_backend/ada.py` 沿用旧文件名，实际模型由配置决定，当前冻结方法使用 BGE-M3。

## 附加题一：检索方法改进

当前混合检索 dev Hit@10=0.44、Hit@1=0.16，抽取式答案 AnswerEM=0。可从检索、重排、查询改写或多轮 Agent 中选择改进方向，并与已有完整 dev50 对照。

- 为新方法单独保存配置和输出，保留 `configs/final.json` 作为基础对照。
- 向量检索器实现 `retrieval.interfaces.VectorRetriever`，融合方法实现 `DocumentFusion`，通过 `module:factory` 加载，见[向量接口](../VECTOR_PLUGIN_GUIDE.md)。
- 固定流程使用 `retrieval` 日志；模型自主选择工具时使用 `agent` 与 `model_tool`，通过 `run.py --method-type agent --model-label <实际模型>` 运行。
- 先在 train 冒烟，再评测完整 dev50，报告检索指标、答案、耗时、token 和失败分布。参数冻结后运行 test50。

生成式答案客户端和 `configs/model.example.json` 可复用，尚无该方案的正式实验结果。

## 附加题二：小模型后训练

已完成 train100 教师轨迹采集、按题分组的数据处理、助手动作监督 LoRA、完整 dev50 对照和冻结 test50。运行环境为 macOS arm64；采集与训练命令见[后训练复现](POSTTRAINING.md)，冻结权重在 [`weights/posttraining/`](../weights/posttraining/)。

## 数据与产物交接

| 内容 | 路径 / 获取方式 |
| --- | --- |
| 课程语料、问题与标签 | `corpus/`、`datas/`，从课程渠道获取 |
| 切块与 BM25 索引 | `artifacts/chunks/`、`artifacts/bm25/`，按 README 重建 |
| BGE-M3 索引与缓存 | `artifacts/bge_m3/`，API 重建或课程内部传递 |
| 基础 dev50 | `output/hybrid/bge_hybrid_dev_20261103/` |
| 基础最终 test50 | `output/hybrid/test_base_final/` |
| 后训练教师、dev/test、训练样本与日志 | 由附加题二成员的课程交付包提供，目录见[后训练复现](POSTTRAINING.md#文件与交付) |
| 聚合实验记录 | Git 中的 `records/bm25/`、`records/hybrid/`、`records/posttraining/` |
| 冻结 LoRA 适配器 | Git 中的 `weights/posttraining/` |
| API 密钥 | 自行设置环境变量 `OPENROUTER_API_KEY` |

最终课程压缩包包含源码、依赖、配置、运行入口、实验报告，以及完整 test50 的 `results.jsonl`、`run_info.json` 和 `traces/`。附加题二另附数据处理和训练脚本、训练配置、数据统计与适配器；其内部交付包还保存训练样本、教师轨迹、dev 对照和完整案例。语料、标签、基础模型和缓存不重复打包。

文档 ID、字符跨度、日志和预算遵循 [SPEC.md](../SPEC.md)。完整问答、原文和逐题轨迹按 [DATA_USE.md](../DATA_USE.md) 在课程内部流转。
