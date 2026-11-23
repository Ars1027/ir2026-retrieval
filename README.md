# IR 2026：检索与小模型后训练

现代信息检索课程大作业，在固定的 10,000 篇文档上检索、回答问题并提供原文证据。数据划分为 train100、dev50、test50。

基础方法为 **BM25 + BGE-M3 + 文档级 RRF**，冻结配置是 `configs/final.json`。附加题一比较 BM25 候选内的段落重排与消融。附加题二已完成教师轨迹采集、Qwen2.5-1.5B LoRA 训练和训练前后对照。

## 实验结果

| 完整 dev50 | BM25 | BM25 + BGE-M3 + RRF | 小模型训练前 | 小模型 LoRA 后 |
| --- | ---: | ---: | ---: | ---: |
| Hit@1 | 0.20 | 0.16 | 0.00 | 0.08 |
| Hit@5 | 0.30 | 0.34 | 0.02 | 0.16 |
| Hit@10 | 0.40 | 0.44 | 0.04 | 0.22 |
| EvidenceRecall@10 | 0.199889 | 0.200857 | 0.020000 | 0.076246 |
| AnswerEM | 0.00 | 0.00 | 0.00 | 0.04 |
| 正常完成 / 50 | 50 | 50 | 22 | 28 |
| 平均每题秒数 | 1.7549 | 5.9743 | 7.9095 | 20.3208 |

小模型两列使用相同的 BM25 工具 Agent，仅适配器不同。基础混合检索采用固定流水线，继续作为基础任务提交方案。全部失败题计入指标分母。

两套最终 test50 均通过官方提交检查：基础混合检索 50 条 `ok`；后训练模型 23 条 `ok`、27 条 `budget_exceeded`。测试标签由教师持有，测试效果由教师统一评测。

## 基础检索

使用 Python 3.13.12 验证。将课程提供的 `corpus/`、`datas/` 放到项目根目录，然后安装依赖并构建索引：

```bash
python -m pip install -r requirements.txt
python verify_data.py --full
python scripts/build_chunks.py
python scripts/build_bm25.py

# BM25 对照，无需 API 密钥
python scripts/run_baseline.py --method bm25 --split dev --output output/bm25_dev_new --check --evaluate
```

混合检索通过 OpenRouter 的 `baai/bge-m3` 编码语料和查询。设置环境变量 `OPENROUTER_API_KEY` 后运行：

```bash
python scripts/build_vector.py --config configs/vector.bge.json --workers 8
python scripts/run_baseline.py --method final --split dev --output output/hybrid_dev_new --check --evaluate
python scripts/run_baseline.py --method final --split test --output output/hybrid_test_new --check
```

Windows 可使用 `./run.ps1`，其他平台可使用 `bash run.sh`，参数与 `scripts/run_baseline.py` 相同。输出目录使用新名称。已有测试结果可单独检查：

```bash
python check_results.py --split test --run-dir output/hybrid/test_base_final --submission
```

## 附加题一：段落 BM25 重排

在相同的词块 BM25 候选上，基础、等权融合和仅段落排序的 dev50 Hit@10 分别为 0.40、0.42、0.42。三组复用现有的抽取式答案、引用和运行入口。运行命令、失败案例、时间开销和逐题配对统计见[段落重排实验](docs/RERANKING.md)。

## 附加题二：小模型 LoRA

运行环境为 **Apple Silicon、macOS arm64、Python 3.12**，实验机器有 32 GB 统一内存。教师为 Qwen3-4B，学生为 Qwen2.5-1.5B，两者使用固定 revision 的本地 MLX 4-bit 权重。

从 train100 筛选得到 21 题、78 条唯一步骤样本，其中 20 题用于梯度训练。最终训练后 4 层 LoRA、rank=8、学习率 2e-5、120 次更新。冻结适配器位于 [`weights/posttraining/`](weights/posttraining/)。

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/posttraining.txt
bash run_bonus2.sh models

# 先按上节构建 BM25 索引，再加载仓库中的冻结适配器运行
bash run_bonus2.sh run --config configs/posttraining/student_frozen.json \
  --split dev --output output/posttraining/frozen_dev_new
```

采集、训练和完整对照命令见[后训练复现说明](docs/POSTTRAINING.md)。后训练提高了文档命中率，同时增加耗时与无效动作；训练数据规模、行为案例和成本见[后训练实验结果](docs/POSTTRAINING_RESULTS.md)。

## 文档与目录

- [基础实验结果](docs/RESULTS.md)：dev/test、构建资源和历史实验。
- [项目交接](docs/HANDOFF.md)：模块入口、附加题实验与课程交付文件。
- [后训练复现](docs/POSTTRAINING.md)与[后训练结果](docs/POSTTRAINING_RESULTS.md)：附加题二实现与实验。
- [向量接口](VECTOR_PLUGIN_GUIDE.md)：自定义检索器与融合方法。
- [课程协议](SPEC.md)与[数据使用约定](DATA_USE.md)。

`retrieval/` 负责切块、检索与排名，`pipeline/` 组装证据和答案，`posttraining/` 实现教师与学生 Agent。配置在 `configs/`，聚合实验记录在 `records/`，冻结适配器在 `weights/`。课程数据、索引、完整逐题结果和训练样本通过课程内部渠道交接，路径见[项目交接](docs/HANDOFF.md)。

## 测试

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests/core -v
python -m unittest discover -s tests/pipeline -v
python -m unittest discover -s tests/vector_backend -v
# 后训练测试在上述 macOS 环境执行
python -m unittest discover -s tests/posttraining -v
```

记录中的测试共 131 项，包含 3 项段落重排测试与 20 项后训练测试。
