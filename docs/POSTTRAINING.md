# 附加题 2：教师轨迹与小模型 LoRA

本目录实现题目第 16 页的后训练流程：仅在课程 train100 采集教师交互轨迹，以历史观测为输入，监督下一步模型动作或答案；同一学生模型在同一检索流程下比较训练前后。实验结果及适配器校验值见 `records/posttraining/`；冻结适配器放在 `weights/posttraining/`，逐题数据和训练样本保留在本地课程交付包中。

## 方法

教师使用 [Qwen3-4B-Instruct-2507 的 MLX 4-bit 版本](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit)，学生使用 [Qwen2.5-1.5B-Instruct 的 MLX 4-bit 版本](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit)。[原始学生模型](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)为 1.54B 参数。两个下载版本固定在各自配置的 commit SHA，`models/*/provenance.json` 保存文件摘要。所有轨迹采集、训练和推理都在本机进行。

教师、未训练学生和训练后学生共用 `posttraining/agent.py`。模型每轮输出一条 JSON 动作：`search`、`read` 或 `finish`。搜索复用原项目的 BM25 切块索引，每次给模型最多 8 篇文档的片段；读取只能打开已观察的文档，每次最多 1,000 字符。模型决定下一次搜索词、读取位置、终止时机、文档排序和短答案。最终引用直接复制实际返回给模型的原文片段，不让模型重写引用。

统一使用 greedy 解码、10 次工具调用、12 个模型回合、每次生成最多 256 tokens，且保留课程的 600 秒硬超时。搜索和读取是实际 `model_tool` 事件；格式错误计入模型调用和 token，不伪造一次工具调用。达到预算或发生错误时，保留最新实际检索排名及原文引用，并保留相应失败状态。`fallback_ranking` 单独记录这种情况。

## 训练数据与损失

`collect` 的划分固定为 train，没有 dev/test 采集选项。推理模块只读取题目、语料和索引，不加载标签。离线预处理阶段才读取 train 标签，过滤非 `ok` 或没有命中 gold 文档的整条轨迹；GoldHit 筛选不等同于答案语义正确。所有被过滤轨迹仍保存在原始采集目录中，并计入教师完整训练集评测。

原始训练题按固定种子 2026 的哈希排序，先按题划出 10% 作为内部验证，再过滤、展开成逐步样本。同一道题的不同回合不会分散到训练和内部验证两侧。课程 dev50 只用于模型完整对照评测；test 只在方法冻结后运行。

每条样本保存教师作出该动作前的**实际输入 messages**，追加该次教师输出。使用 [MLX-LM 的 LoRA 训练](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)，`mask_prompt=true`，只计算最后一条 assistant 消息的损失。此前的模型动作、错误提示和工具返回均不作为生成目标。预处理逐条检查 chat-template 前缀和实际 token 长度，超长样本过滤并统计，不静默截断监督目标。

配置位于 `configs/posttraining/lora.yaml`：4-bit 基础权重冻结，后 16 层 LoRA，rank=8、scale=16、dropout=0.05，AdamW、学习率 1e-4、batch=1、240 次更新、最大序列 4096、梯度检查点、seed=2026。固定使用最后一次更新的适配器；没有根据测试反馈选择 checkpoint。MLX 的验证损失在对应编号更新**之前**计算，报告明确这一时序。

## 复现

已验证环境为 Apple Silicon、32 GB 统一内存、Python 3.12。此可选训练后端面向 macOS arm64；基础任务的依赖和配置独立保留。

把课程包中的 `corpus/`、`datas/` 放在项目根目录，也可以建立指向课程数据的本地符号链接。所有命令从项目根目录执行，输出目录须使用新名称。

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

`run` 调用原课程 `run.py`，每题独立子进程运行并由父进程执行硬超时。模型每题加载时间包含在单题耗时中。每次运行自动执行 `check_results.py`；train/dev 还执行 `evaluate.py`。完整划分使用 `--submission` 检查。test 配置必须包含已训练适配器的冻结摘要，适配器变化会拒绝运行。

对照汇总不仅比较配置，还核对实际提示词、BM25 文件和基础权重摘要。必须完整覆盖 dev50，失败也留在分母中；报告 Hit@1/5/10、EvidenceRecall@10、AnswerEM、状态、耗时、模型调用、工具调用与 token。token 统计来自实际本地生成器；中断或失败生成无法取得完整统计时使用 `null`。本地推理的 API 调用费用为 0，电费未计量。

```bash
.venv/bin/python -m unittest discover -s tests/posttraining -v
```

## 文件与交付边界

| 内容 | 路径 |
| --- | --- |
| 教师/学生/训练配置 | `configs/posttraining/` |
| 采集、预处理、训练、运行、汇总统一入口 | `scripts/posttrain.py`、`run_bonus2.sh` |
| 动作协议、实际观测和推理 | `posttraining/policy.py`、`tools.py`、`agent.py` |
| 完整轨迹和教师逐步决策 | `output/posttraining/teacher_train/` |
| SFT 样本、来源和数据统计 | `artifacts/posttraining/data/` |
| LoRA 权重、配置和训练日志 | `artifacts/posttraining/adapter/` |
| 可直接检查和加载的冻结适配器 | `weights/posttraining/` |
| 训练前后逐题输出 | `output/posttraining/student_*_dev/` |
| 公开的聚合记录 | `records/posttraining/` |

遵循 [DATA_USE.md](../DATA_USE.md)，公开 PR 包含代码、配置、合成测试、聚合结果和冻结 LoRA 适配器；不公开反混淆问答、文档原文、逐题决策或搜索轨迹。本地课程交付包另含数据统计、完整验证/测试结果与日志、训练数据和带案例的报告；不重复打包课程语料、标签、基础模型或缓存。
