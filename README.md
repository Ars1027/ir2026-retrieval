# 现代信息检索大作业

给定自然语言问题，从 BrowseComp-Plus 课程子集的 10,000 篇文档中返回最多 10 篇相关文档、5 段原文引用和一个简短答案。基础任务提交 **BM25 + BGE-M3 文档级 RRF**。在完整 dev50 上，纯 BM25 命中 20 题，混合检索命中 22 题。

| 实验 | 配置 | 完整 dev50 Hit@10 |
| --- | --- | ---: |
| 基础任务，BM25 / 混合检索 | `configs/bm25.json`、`configs/final.json` | 0.40 / 0.44 |
| 附加题一，段落融合 / 仅段落 | `configs/bonus1/{fusion,passage}.json` | 0.42 / 0.42 |
| 附加题二，同一学生模型训练前 / LoRA 后 | `configs/posttraining/{student_before,student_frozen}.json` | 0.04 / 0.22 |

## 基础检索

从项目根目录运行。将课程包的 `corpus/` 和 `datas/` 放在根目录，分别提供原文与问题划分。基础实验使用 Python 3.13.12、NumPy 2.4.4 和 faiss-cpu 1.13.0。设置环境变量 `OPENROUTER_API_KEY` 后可构建 BGE-M3 索引和编码查询。纯 BM25 与附加题一支持离线运行。

```bash
python -m pip install -r requirements.txt
python verify_data.py --full
python run.py --split train --limit 3 --agent examples.keyword_agent --output output/course_demo3
python check_results.py --split train --run-dir output/course_demo3 --questions output/course_demo3/questions.jsonl
python scripts/build_chunks.py
python scripts/build_bm25.py
python scripts/build_vector.py --config configs/vector.bge.json --workers 8
```

切块和两个索引分别写到 `artifacts/chunks/`、`artifacts/bm25/` 和 `artifacts/bge_m3/`。下面的命令依次运行离线三题检查、完整 dev 对照、冻结配置 test50。每次运行使用新的输出目录。

```bash
bash run.sh --method bm25 --split train --limit 3 --output output/smoke3 --check
bash run.sh --method bm25 --split dev --output output/bm25_dev --check --evaluate
bash run.sh --method final --split dev --output output/hybrid_dev --check --evaluate
bash run.sh --method final --split test --output output/hybrid_test --check
```

Windows 下将上述 `bash run.sh` 换成 `./run.ps1`。每个运行目录包含逐题排名、答案和证据 `results.jsonl`、运行参数与预算 `run_info.json`、逐题日志 `traces/<id>.jsonl`。dev 加上 `--evaluate` 会生成 `metrics.json` 和 `per_query.csv`。

每题独立运行，限时 600 秒。模型工具交互最多 32 次，每次观测最多 12,000 字符。超时、调用超限、异常及空答案分别记录状态。评测以完整划分为分母。固定检索流程将召回、融合、取证与答案写为检索事件，BGE 输入 token 单独计量。

## 附加题一：段落 BM25 重排

在前 200 个 BM25 词块中取最多 100 篇候选文档，分别运行段落融合和仅段落排序。两组共用原文切窗、证据和答案抽取流程。

```bash
bash run.sh --method bm25 --config configs/bonus1/fusion.json --split dev --output output/fusion_dev --check --evaluate
bash run.sh --method bm25 --config configs/bonus1/passage.json --split dev --output output/passage_dev --check --evaluate
bash run.sh --method bm25 --config configs/bonus1/fusion.json --split test --output output/fusion_test --check
```

[段落重排实验与四个案例](docs/RERANKING.md)

## 附加题二：小模型 LoRA

使用 Apple Silicon、macOS arm64、Python 3.12。实验机为 32 GB 统一内存。先构建上述 BM25 索引，再下载配置指定 revision 的本地教师和学生模型。训练前后共用 BM25 工具、提示词、greedy 解码和预算。

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/posttraining.txt
bash run.sh bonus2 models
bash run.sh bonus2 run --config configs/posttraining/student_before.json --split dev --output output/student_before_dev
bash run.sh bonus2 run --config configs/posttraining/student_frozen.json --split dev --output output/student_after_dev
bash run.sh bonus2 run --config configs/posttraining/student_frozen.json --split test --output output/student_test
```

`run` 自动检查完整结果并评测有标签的划分。教师采集、数据处理、训练和对照使用同一入口的 `collect`、`prepare`、`rebalance`、`train` 和 `compare` 子命令。[训练步骤](docs/POSTTRAINING.md) · [训练结果与工具行为](docs/POSTTRAINING_RESULTS.md)

## 课程提交包

| 内容 | 包内路径 |
| --- | --- |
| 实验报告、可编辑排版源文件 | `report/实验报告.pdf`、`report/实验报告.tex` |
| 基础任务正式 test50 排名、引用、参数、逐题日志 | `output/hybrid/test_base_final/` |
| 附加题一冻结融合版 test50 | `output/bonus1/test_fusion/` |
| 附加题二数据处理、训练入口和配置 | `posttraining/`、`scripts/posttrain.py`、`configs/posttraining/` |
| 后训练数据统计、训练日志摘要、开发与测试统计 | `records/posttraining/` |
| 最终 LoRA 权重和加载配置 | `weights/posttraining/` |

核验包内结果时，先放入课程数据，再执行：

```bash
python check_results.py --split test --run-dir output/hybrid/test_base_final --submission
python check_results.py --split test --run-dir output/bonus1/test_fusion --submission
```

## 来源与实现分工

课程发布包提供数据协议、运行器、检查器和评测脚本。小组实现 `retrieval/` 的分块、倒排索引和段落排序，`vector_backend/` 的 BGE-M3 编码、FAISS 检索和融合，`pipeline/` 的引用与答案组装，以及 `posttraining/` 的轨迹处理、动作监督和训练评测。

数据来自 [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)，版本与许可见 [DATA_USE.md](DATA_USE.md)。向量检索使用 [BAAI BGE-M3](https://huggingface.co/BAAI/bge-m3) 和 [FAISS](https://github.com/facebookresearch/faiss)。本地后训练使用 [MLX-LM](https://github.com/ml-explore/mlx-lm) 和 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，适配器目录附基座许可。
