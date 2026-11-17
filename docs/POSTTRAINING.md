# 附加题二：教师轨迹与小模型 LoRA

从 train100 采集教师交互轨迹，以历史观测预测下一步动作或答案，再比较同一学生训练前后的完整 dev50。实验指标、案例和训练配置见[后训练结果](POSTTRAINING_RESULTS.md)，冻结适配器在 [`weights/posttraining/`](../weights/posttraining/)。

## 方法与训练数据

教师为 [Qwen3-4B-Instruct-2507 MLX 4-bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit)，学生为 [Qwen2.5-1.5B-Instruct MLX 4-bit](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit)，原始学生为 1.54B 参数。两者在配置中固定 revision，下载文件摘要保存在 `models/*/provenance.json`。

教师与训练前后学生共用 `posttraining/agent.py`，每轮输出一条 JSON 动作：`search`、`read` 或 `finish`。搜索返回最多 8 篇文档的 BM25 片段；读取限于已观察文档，每次最多 1,000 字符；模型决定查询词、读取位置、终止、排名和短答案。引用复制实际观测的原文片段。

统一采用 greedy 解码、10 次工具调用、12 个模型回合、每次最多生成 256 tokens，以及课程的 600 秒硬超时。搜索与读取记录为 `model_tool`；格式错误计入模型调用和 token。预算失败或错误保留最新检索排名、引用和失败状态，并记录 `fallback_ranking`。

`collect` 只采集 train100。离线预处理保留 `ok` 且命中 gold 文档的轨迹；终止样本还要求标准答案能在当前可见的 gold 原文中按词边界逐字核验。搜索和读取目标沿用教师输出；错误终止答案可用训练标签校正，来源记为 `visible_gold_train_label_correction`。原始决策保留，推理模块读取题目、语料和索引。

按 seed=2026 的题目哈希排序，先划出 10% 内部验证，再过滤并展开步骤。同一道题的所有步骤属于同一划分。dev 用于选参数，test 在冻结后运行。

训练样本保存动作前的实际 messages，并追加目标助手输出。使用 [MLX-LM LoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) 的 `mask_prompt=true`，仅对最后一条助手消息计算损失，历史动作和工具返回只作输入。预处理检查模板前缀和 token 长度，超长样本过滤并统计。

最终配置为 `configs/posttraining/lora.yaml`：冻结 4-bit 基座，后 4 层 LoRA，rank=8、scale=8、dropout=0.05，AdamW、学习率 2e-5、batch=1、120 次更新、最大序列 4096、梯度检查点、seed=2026。终止样本仅在训练侧重复采样 3 次。初始 240 步方案及其筛查结果见[实验结果](POSTTRAINING_RESULTS.md#数据与实际训练)。MLX 验证损失在对应编号更新之前计算。

## 复现

环境为 **Apple Silicon、macOS arm64、Python 3.12**，实验机器有 32 GB 统一内存。把课程 `corpus/`、`datas/` 放在项目根目录或建立符号链接；命令从项目根目录执行，输出目录使用新名称。

下面从采集重建完整实验。直接加载公开冻结权重的命令见 [README](../README.md#附加题二小模型-lora)。

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/posttraining.txt
.venv/bin/python verify_data.py --full
.venv/bin/python scripts/build_chunks.py --config configs/bm25.json
.venv/bin/python scripts/build_bm25.py --config configs/bm25.json

# 下载固定版本的本地模型，运行阶段无需模型 API 密钥。
bash run_bonus2.sh models

# 先 train 冒烟；此目录不能作为完整训练数据来源。
bash run_bonus2.sh collect --limit 3 --output output/posttraining/teacher_smoke_new
bash run_bonus2.sh collect --output output/posttraining/teacher_train_new
bash run_bonus2.sh prepare --teacher-run output/posttraining/teacher_train_new
bash run_bonus2.sh rebalance

# 同一模型训练前的完整 dev50。
bash run_bonus2.sh run --config configs/posttraining/student_before.json \
  --split dev --output output/posttraining/student_before_dev_new
bash run_bonus2.sh train
bash run_bonus2.sh run --config configs/posttraining/student_after.json \
  --split dev --output output/posttraining/student_after_dev_new
bash run_bonus2.sh compare --before output/posttraining/student_before_dev_new \
  --after output/posttraining/student_after_dev_new
bash run_bonus2.sh plot

# 开始最终 test 前，固定适配器摘要。全部 50 题，不提供子集参数。
bash run_bonus2.sh freeze
bash run_bonus2.sh run --config configs/posttraining/student_frozen.json \
  --split test --output output/posttraining/student_test_new
```

`run` 调用课程 `run.py`，每题使用独立子进程，加载模型计入单题耗时。运行后自动检查结果，train/dev 还执行评测；完整划分使用 `--submission`。test 要求冻结配置及匹配的适配器摘要。

### 中断续跑

当前运行器在整批结束后写入资源摘要；中断运行使用 `--resume` 时，需通过 `--resource-reference` 提供条件匹配的完整运行记录：

```bash
bash run_bonus2.sh run --config configs/posttraining/student_after.json \
  --split dev --output output/posttraining/student_after_dev_final --resume \
  --resource-reference output/posttraining/student_before_dev_v2/run_info.json
```

续跑核对问题序列、已完成结果、模型、配置及资源摘要。已完成的失败题保留，只重跑尚无结果的题目；中断尝试的决策和观测存入 `interrupted_attempts/`，接续记录写入 `resume_history.jsonl`。单题耗时不含暂停和中断尝试的额外计算。

对照汇总核对配置、提示词、BM25 文件和基础权重摘要，报告全部 dev50 的指标、状态、成本和行为统计。token 来自本地生成器，包含重复历史；生成中断导致用量未知时记为 `null`。本地模型 API 费用为 0，电费未计量。

```bash
.venv/bin/python -m unittest discover -s tests/posttraining -v
```

## 文件与交付

| 内容 | 路径 |
| --- | --- |
| 教师、学生、训练配置 | `configs/posttraining/` |
| 采集、预处理、训练、评测与打包入口 | `scripts/posttrain.py`、`run_bonus2.sh` |
| 教师完整轨迹和逐步决策 | `output/posttraining/teacher_train/` |
| 唯一 SFT 样本与来源 | `artifacts/posttraining/data/` |
| 采样后的训练数据 | `artifacts/posttraining/data_balanced/` |
| 最终训练权重、日志和摘要 | `artifacts/posttraining/adapter_conservative/` |
| 初始 240 步训练产物 | `artifacts/posttraining/adapter/` |
| 公开冻结适配器与加载配置 | `weights/posttraining/` |
| 训练前后完整 dev50 | `output/posttraining/student_before_dev_v2/`、`output/posttraining/student_after_dev_final/` |
| 冻结 test50 | `output/posttraining/student_test_final/` |
| 聚合记录、训练曲线 | `records/posttraining/` |

上述完整实验产物由成员的课程交付包提供，本地重新运行时以所选输出目录为准。按 [DATA_USE.md](../DATA_USE.md)，代码、配置、合成测试、聚合记录和适配器公开；完整问答、训练样本、原文和逐题轨迹在课程内部流转。

在原始训练产物和各次完整运行目录齐备时，可生成内部交付包及含四个完整案例的 `BONUS2_REPORT.md`：

```bash
bash run_bonus2.sh package \
  --before output/posttraining/student_before_dev_v2 \
  --after output/posttraining/student_after_dev_final \
  --test output/posttraining/student_test_final \
  --teacher output/posttraining/teacher_train \
  --output output/posttraining/bonus2_delivery.zip
```

交付包包含源码、冻结适配器、训练样本与日志、完整运行结果及报告。课程语料、标签、基础模型和缓存另行准备。
