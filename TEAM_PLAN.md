# 两人开发分工：基础任务全部实现、测试与初版结果

范围确认：截至两个加分项开始之前。包含数据接入、自己的检索方法、证据与答案、批量运行、检查评测、初版结果、成本记录和复现入口；不包含课程报告、汇报材料、案例分析文章、多轮搜索策略改进、教师轨迹采集、SFT/LoRA训练。

角色暂用 A＝你、B＝另一位同学。两人都开发检索、都写自己模块的测试、都运行实验；没有专职测试角色。姓名和 GitHub 用户名确定后再替换，不预先邀请未知账号。

## 0. 当前交付状态与使用方式

- 本仓库以下载的课程项目为起点，保留上游 README、SPEC、评测器、运行器、示例与许可说明。
- `agent/solution.py` 仍为上游未实现模板。下文列出的新模块、命令和任务是待完成计划，不能把建仓视为检索方案已实现。
- 已完成的初始检查见 `records/initial_checks_*.json`；仅包含环境、测试与官方3题演示的汇总数据，不代表自己的 BM25/混合检索基线成绩。
- 本阶段交付两套自己的方法：`bm25-v1` 和 `hybrid-v1`。混合检索作为基础系统的较强版本，本阶段不据此申报加分项。
- 公共仓库只提交源代码、配置、合成测试、汇总指标与本计划。真实问题、标签、原文、原文引用、逐题答案和轨迹留在本地。

## 1. 技术方案与边界

共同数据流：原文 → 保留原文位置的切块 → BM25 / 向量两路召回 → 文档级去重与排序 → 选择原文证据 → 可选固定答案生成 → 课程接口输出。

1. A 实现 BM25，B 实现 dense，二者返回同一种片段候选结构。
2. 先完成自己的 BM25 端到端版本，再将 dense 和 RRF 接入形成 hybrid 版本。
3. 使用固定流水线，`method_type=retrieval`；向量编码或一次答案生成使用模型也仍可为 retrieval，必须声明实际模型和成本。
4. 本阶段不做 Agent 自主工具选择、多轮证据反馈、重排模型训练或学生模型微调；只预留函数接口，不编写空壳训练系统。
5. BM25 用成熟的倒排索引实现，具体库以兼容环境并完成小样本验证后锁定；不要每题对所有片段执行 Python 全量词项循环。
6. dense 初选 `Qwen/Qwen3-Embedding-0.6B`、sentence-transformers、FAISS；按模型要求设置查询指令与归一化。若硬件不足，可用 CPU 或更换模型，必须记入配置并重新测量。当前不承诺特定显存能跑下所有配置。
7. RRF 按名次融合，不直接相加 BM25 原始分数和余弦相似度。先在两路内将片段聚合成文档排名，再对文档排名融合；文档保留原始片段供证据选择。
8. 默认每路候选片段先取50；RRF常数60；切块约512 token、重叠约64 token，均只是开发起点。A先实现可复现切块；B的编码器不得二次截断造成位置失真。最终参数由验证集选择并冻结。
9. 同一原文在多处出现相关片段时，最终文档ID仅占一个位置；最多10篇文档，最多5段引用，不要求为10篇文档各提供一段引用。

参考实现资料（非性能承诺）：
- [Qwen3-Embedding-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [RRF 说明](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- 本地协议以 `SPEC.md`、`configs/default.json` 和实际校验器为准。

## 2. 先一起冻结的接口（M0，约半天）

共同编写 `retrieval/types.py`、`configs/baseline.json` 和一个仅用虚构文本的 fixture，A提交初版、B审阅。不要各自定义不同字段。

```python
# Chunk：start/end 为 Corpus.text(doc_id) 返回文本的 Python 字符位置。
{"chunk_id": "opaque-id", "doc_id": "original-string-id", "start": 0,
 "end": 240, "text": "original_text[start:end]"}

# Hit：score 只在同一种召回器内解释，不能跨召回器直接比较。
{"chunk_id": "opaque-id", "doc_id": "original-string-id", "start": 0,
 "end": 240, "score": 1.23, "rank": 1, "source": "bm25"}

# 每路检索器共同提供：
build_index(corpus_root, index_dir, settings)  # 离线、只依赖语料与配置
load_index(index_dir, settings)
search(query: str, top_k: int) -> list[Hit]

# 共享输出接口：
select_evidence(question, ranked_docs, hit_spans, corpus) -> list[dict]
answer_from_evidence(question, evidence, model_config) -> dict
```

`doc_id` 始终是课程原始字符串ID。检索模块不导入 train/dev 标签或 evaluate；标签只在离线评测进程读取。所有模块以参数接收路径，不硬编码当前用户的绝对路径。实际 `solve(question, tools, config)` 的传参方式保持课程接口兼容。

## 3. A 的任务：关键词检索＋证据和答案（预计约8—11个有效人日）

### A1 数据适配与保真切块（1—1.5日）

- 文件：`retrieval/corpus_adapter.py`、`retrieval/chunking.py`、`scripts/build_chunks.py`。
- 基于 `Corpus` 或相同清单读取文档；保留原始ID、段落边界及字符位置。检索用的大小写归一化和分词结果另存，不覆盖原文。
- 切块产物放 `artifacts/chunks/`，不进Git；记录语料清单摘要、切块参数、tokenizer版本、片段数及构建时间。
- 测试：Unicode、中英文、空段落、极长段落、结尾不足一块、重叠、重复正文；逐块验证 `chunk.text == corpus.text(doc_id)[start:end]`。
- 验收：10000篇文档全部处理，非空正文有片段；相同输入配置能重建同样的ID和边界；B能直接拿切块产物建向量索引。
- 依赖：M0；交接B：fixture当天交、完整产物随后交。

### A2 BM25 索引与检索（1.5—2日）

- 文件：`retrieval/bm25.py`、`scripts/build_bm25.py`、`tests/test_bm25.py`。
- 实现分词、索引持久化、加载和 `search`；人名、年份、连字符等不要简单清洗掉。稳定处理同分排序。
- 返回片段候选及原始位置，不在检索器里生成答案。词面统计只来自语料，不使用问题标签构造索引。
- 记录索引大小、构建时间、加载时间、查询耗时；避免每题重新切块和建索引。
- 验收：合成语料的精确实体/年份查询符合预期；重启加载与初次检索排名一致；空查询和无命中有确定返回。
- 依赖：A1；交接B：统一Hit列表，供融合接入。

### A3 文档聚合与证据定位（1—1.5日）

- 文件：`retrieval/document_rank.py`、`pipeline/evidence.py`、对应合成测试。
- BM25和dense都复用同一个片段→文档聚合逻辑；初版采用每文档最高排名片段，稳定去重，保留其他候选片段映射。
- 根据问题与片段相关性选择原文连续跨度；必要时扩展到句子或邻近上下文，不能拼接多处正文当一条引用。
- 每段20—1200字符、最多5段；引用文档须在最终列表内，且至少有一段引用才能返回文档。
- 验收：每段引用在 `Corpus.text()` 中逐字存在；没有未知ID、重复文档或越界；B的dense候选也可直接使用。

### A4 固定答案生成接口（1—1.5日）

- 文件：`pipeline/answer.py`、`pipeline/model_client.py`、`configs/model.example.json`。
- 输入只包含问题和已选原文证据；输出简短答案和实际模型usage；不启用联网搜索，不让模型自行访问文件。
- 实现可选模型模式：没有凭证或证据不足时如实 `no_answer`，仍保留有效检索结果。无模型模式不能伪称完整问答能力已交付。
- 模型来源、版本、温度、超时、最大输出和token口径配置化；密钥只从环境变量取，不写日志与Git。
- 不让模型生成最终引用文本：引用由已验证的原文跨度形成，模型只作答或选择证据ID。
- 验收：有证据可答、证据不足、模型异常三类合成场景可处理；不把缺失token写成0；API超时后保留检索成果。
- 若双方决定基础系统不用生成模型，本任务以明确的抽取式答案策略替代；不能仅返回原文片段冒充短答案。变更要写入冻结配置。

### A5 BM25 端到端接入与本人实验（1—1.5日）

- 文件：`agent/bm25_solution.py`，调用共享证据/答案模块。
- 用 `tools.record_retrieval()` 记录真实的召回、文档聚合和证据定位阶段，不能只补最终ID伪造过程。
- 先训练集3—5题冒烟，再验证集完整50题；运行官方检查与评测，保存本地逐题结果与轨迹。
- 将汇总指标交给B的汇总脚本，记录召回成本和答案生成成本；未运行项填null，不估算成实测。
- 验收：自研方法完整50题通过 `--submission` 检查；该检查的通过不代表答案语义正确。

### A6 交叉检索与冻结复核（约1日）

- 审阅B的dense查询指令、向量归一化和ID映射；使用同一fixture核对RRF次序。
- 在B的分支上实际运行hybrid冒烟，确认不是只有B自己的机器能用。
- 和B共同核对至少4条真实结果（成功/失败均有），仅留结构化本地记录：召回遗漏/排序靠后/证据错位/答案错误/资源失败。此项是开发排错，不撰写案例分析文章。
- 测试集只在全部配置冻结后参与一次最终运行的交叉检查，不据此调参。

## 4. B 的任务：向量检索＋融合排序和运行集成（预计约8—11个有效人日）

### B1 向量编码与资源摸底（1—1.5日，可与A1并行）

- 文件：`retrieval/embedding.py`、`configs/dense.json`、合成编码fixture。
- 先用虚构文本搭通encode接口，再接A的切块；区分query与document编码，固定模型revision、指令、维度、batch size、最大长度与归一化。
- 检查显卡/CPU、内存和模型兼容性，实测一个小批次；选择能稳定运行的配置后固定依赖，保留随机种子及设备信息。
- 原文离线编码只做一次；不将整个原文超过模型长度后静默截掉。
- 验收：同义表达合成例能召回目标片段，向量维度/数值正常；无GPU时给出已验证的CPU配置或明确资源缺口。

### B2 FAISS 索引与dense检索（1.5—2日）

- 文件：`retrieval/dense.py`、`scripts/build_dense.py`、`tests/test_dense.py`。
- 与A共同维护片段ID映射。初版优先用归一化向量＋精确内积检索，实际规模/内存不适合时再采用近似索引并记录参数。
- 构建分批写入向量和索引到 `artifacts/dense/`，保存语料/切块/模型配置摘要；禁止把不同切块版本的向量与ID表混用。
- 测试：索引重载、ID对应、空候选、重复片段、维度不一致和配置不匹配。
- 验收：完整语料编码成功；索引重载后ID和排序一致；构建时间、索引大小、查询编码与搜索耗时均有实测。
- 依赖：B1＋A1产物；A1未完成时使用共同fixture，不等待空转。

### B3 文档级RRF融合（约1日）

- 文件：`retrieval/fusion.py`、`tests/test_fusion.py`。
- 先调用A的共享文档聚合器，再对BM25与dense文档排名计算 `sum(1 / (k + rank))`，rank从1开始。
- 同文档在同一路只贡献一次排名分数；一边为空时仍可用另一边；同分用明确规则稳定排序。
- 保留各路来源排名、融合得分、证据片段映射以便追溯；最终裁到10篇。
- 验收：手算的虚构排名与程序一致；两路重叠与不重叠、单路失败、同分均有有效测试。
- 依赖：A2/B2返回共同Hit；可先用fixture开发。

### B4 固定流水线、配置与运行入口（1.5—2日）

- 文件：`pipeline/retrieve.py`、`agent/hybrid_solution.py`、`agent/solution.py`、`scripts/run_baseline.py`、`run.sh` / `run.ps1`。
- 连接BM25、dense、融合、A的证据与答案模块；`agent/solution.py`最终指向冻结的默认方法，而不是原样保留NotImplementedError。
- 提供统一参数入口：数据划分、方法、方法配置、模型配置、输出目录；区分 `configs/default.json`固定预算与本组可调方法参数，不能放宽课程预算。
- 课程 `run.py`没有方法配置CLI，不能直接杜撰 `--config` 参数；新增入口需要通过明确的加载方式向适配器传递配置，记录最终配置快照。
- 优先复用课程运行器和校验器。Windows下每题spawn新进程，不能依赖跨题全局变量缓存重型模型；测试索引加载开销。若加载占比太高，采用常驻本地编码服务或自建批量器，真实导出同一协议，不能隐藏查询计算时间。
- 固定流水线的模型调用不改成Agent交互；未来若改自主动作，另接 `model_tool`日志，当前 `record_retrieval()`仅支持retrieval模式。
- 验收：干净终端可按统一入口建索引、跑指定划分；无数据时给明确缺失路径；输出目录非空时不覆盖既有实验；密钥不进快照。

### B5 本人dense/hybrid实验与汇总记录（1—1.5日）

- 文件：`scripts/summarize_runs.py`，本地 `output/`，可提交汇总 `records/baselines/*.json`。
- 自己执行dense-only小规模检索诊断，再执行hybrid完整验证50题；复用同一证据/答案策略以减少比较混杂。
- 用官方evaluate输出计算主指标，不能将句段命中率当文档Hit@10；保存原始metrics不手改成绩。
- 汇总BM25与hybrid：Hit@1/5/10、EvidenceRecall@10、AnswerEM、status计数、平均/p50/p95耗时、模型tokens、构建/加载成本。
- 有模型价目和账单才能记录费用；不可得写null＋原因，不把工具次数当费用。记录使用的Git commit、配置摘要、数据版本、硬件和依赖版本。
- 验收：聚合记录可从本地原始结果重算；公开记录不包含真实题目ID、问答、原文引用和逐题分数明细。

### B6 交叉检索复核与最终运行（约1日）

- 审阅A的BM25分词和原文位置；在自己的环境运行BM25冒烟。
- 与A共同冻结方法配置，并运行最终测试50题和官方完整检查；A独立核查覆盖、引用和日志。测试无标签，不计算或编造测试Hit指标。
- 本阶段默认交付最终冻结基础方法的测试结果；若后续继续加分项目，另开新版本，不能利用测试问题/反馈调参。

## 5. 依赖顺序与里程碑

| 里程碑 | A并行工作 | B并行工作 | 完成门槛 |
|---|---|---|---|
| M0 接口（第1个工作日） | 切块/Hit结构初稿 | fixture与结构审阅、设备检查 | 双方用同一fixture调用同一接口 |
| M1 两路召回（第2—4日） | A1/A2切块＋BM25 | B1/B2编码＋dense | 两人各自有自己的检索输出，能持久化重载 |
| M2 端到端（第5—7日） | A3/A4/A5证据、答案、BM25适配器 | B3/B4融合、hybrid适配器与入口 | 两种方法训练集冒烟通过，真实日志可检查 |
| M3 初版结果（第8—9日） | 完整dev BM25运行、核对 | 完整dev hybrid运行、汇总 | 两套50题结果都通过完整检查，配置/commit齐全 |
| M4 交叉复现与冻结（第10—12日） | 复跑对方方法、定位失败 | 复跑对方方法、修入口问题 | 同一冻结配置，两人都能复现流程 |
| M5 基础交付（预留第13—14日） | 检查最终覆盖与日志 | 冻结后test50＋检查 | 本地技术交付完整，可开始加分项 |

这是有效工作日估算，不是自然日保证；GPU编码、API额度和课程安排可能增加等待。每人主要实现约8—11人日，另预留2日联调。不要先让A独自做完检索，再让B开始测试。

## 6. 实验矩阵与数据使用

| 阶段 | 数据 | 方法 | 操作者 | 记录 |
|---|---|---|---|---|
| 初始检查（已运行项见records） | 数据哈希、合成tests、dev前3题演示 | 官方keyword demo | 建仓时执行 | 环境/检查/演示汇总，非本组基线 |
| 开发冒烟 | train前3—5题 | 自研BM25 | A | 结果/trace/失败原因本地保存 |
| 开发冒烟 | 同一组train题 | 自研dense/hybrid | B | 结果/trace/失败原因本地保存 |
| 正式初版 | dev全部50题 | bm25-v1 | A | 官方metrics、逐题结果、运行配置 |
| 正式初版 | dev全部50题 | hybrid-v1 | B | 官方metrics、逐题结果、运行配置 |
| 参数选择 | dev全部50题 | 少量预先列出的参数配置 | 双方 | 保留所有成功和失败运行，不只留最佳值 |
| 最终基础交付 | test全部50题 | 已冻结的基础方法 | B运行、A复核 | 只做格式/覆盖检查，不做无标签效果评测 |

建议参数选择先只动候选数与RRF常数，最多列出3—4套候选，避免验证集上无边界搜索。所有运行不漏题、不删超时/错误题。真实标签不进入检索与问答上下文，不使用上游答案、现成轨迹或联网搜索补答案。

固定预算：每题600秒；最终最多10篇文档；最多5段原文引用，每段20—1200字符。模型自主工具交互才受32次、每次观测12000字符限制；普通retrieval阶段不按工具调用计数。最终统计以真实事件类型为准。

## 7. 测试责任按模块分，不按人划成开发/测试

- A自己写：切块位置、BM25排序、文档去重、证据连续性、模型答复与no_answer降级。
- B自己写：向量ID/维度与重载、RRF手算、配置加载、运行入口和汇总一致性。
- 共同跑：课程33项测试；训练集冒烟；两套dev全量检查；冻结后test全量检查。
- A审B的检索与融合，B审A的检索与证据；真实原文/题目不得复制到测试fixture，fixture使用虚构人物/机构和合成文本。
- 不重写或放宽官方打分器来让结果通过。需要课程代码兼容修复时单独提交、解释原因并保留原始语义。

## 8. 运行命令与最终技术交付

下列前三项为现有课程命令，已可运行：

```powershell
python verify_data.py --full
python -m unittest discover -s tests -v
python scripts/record_initial_checks.py
```

适配器实现完后，正式BM25命令示例（当前模板尚不能完成此方法）：

```powershell
python run.py --split dev --agent agent.bm25_solution --method-type retrieval --method-label bm25-v1 --output output/dev_bm25_v1
python check_results.py --split dev --run-dir output/dev_bm25_v1 --submission
python evaluate.py --split dev --run-dir output/dev_bm25_v1
```

hybrid使用 `agent.hybrid_solution`、独立输出目录，并通过 `--model-label`填写实际embedding模型；若答案也使用模型，标识包括两者。方法参数传递方式由B4实现的统一入口确定，最终命令以实现为准。自研BM25若也生成模型答案，同样必须记录模型标识。完整dev无需 `--allow-subset`，正式检查不能指定演示questions文件。

本地应交付：

```text
output/dev_bm25_v1/     results.jsonl、run_info.json、traces/、metrics.json、per_query.csv
output/dev_hybrid_v1/   同上
output/test_base_final/ results.jsonl、run_info.json、traces/、完整检查结果（无测试metrics）
artifacts/              可重建的切块、索引与编码元数据
records/baselines/      可公开的聚合指标、环境、配置摘要和代码commit
```

结果目录还需本地保存方法配置快照、检查输出、运行终端日志和硬件/依赖信息；API密钥、完整请求头不保存。公开汇总使用明确白名单字段，不直接复制 `results.jsonl`、`questions.jsonl`、`per_query.csv`或原始trace。

基础阶段完成定义：两个检索器都由本人理解并能解释；本组BM25与hybrid各有完整dev结果；默认入口可运行；完整性、引用和日志检查通过；最终test50在配置冻结后完成；双方均能运行对方方法；无模型/答案生成限制如实记录。实验报告、答辩PPT和正式案例分析由其他分工承担，本计划不安排写作任务。

## 9. 公共仓库与协作

- 公开代码地址以Git remote为准；不添加不存在的同学账号。
- 分支：A用 `feat/bm25-evidence`，B用 `feat/dense-fusion`；接口变更先给对方一个最小可运行fixture。
- A主要维护 corpus/chunking/bm25/document_rank/evidence/answer；B主要维护 embedding/dense/fusion/retrieve/入口/汇总。`types.py`、课程适配器入口和依赖由双方协商后修改，减少冲突。
- 两人各自提PR，对方审阅；禁止直接把未跑通的重型依赖升级合入主分支。
- 刚clone时没有课程数据，需各自从课程授权渠道下载相同版本，将 `corpus/` 与 `datas/` 放入仓库根目录，再运行全量校验。公开仓库不能独立提供课程数据。
- `.gitignore`不追踪数据、模型、索引、输出、凭证。真实QA和轨迹不上传公开GitHub，来源见上游 `DATA_USE.md`。
- M4以后切标签 `base-v1`，再为两个加分项开新分支；当前只保留检索、读取、答案与真实事件日志的接口，不实施加分任务。
