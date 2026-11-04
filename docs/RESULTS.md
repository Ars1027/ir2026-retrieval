# 基础实验结果

实验使用课程 IR2026-BCP-200-v1：10,000 篇文档，train/dev/test 分别为 100/50/50 题。数据协议和指标定义见 [SPEC.md](../SPEC.md)。

## 完整 dev50 对照

| 指标 | BM25 复跑 | BGE-M3 混合检索 |
| --- | ---: | ---: |
| Hit@1 | 0.20 | 0.16 |
| Hit@5 | 0.30 | 0.34 |
| Hit@10 | 0.40 | 0.44 |
| EvidenceRecall@10 | 0.199889 | 0.200857 |
| AnswerEM | 0.00 | 0.00 |
| ok / 题目总数 | 50 / 50 | 50 / 50 |
| 平均每题耗时（秒） | 1.7549 | 5.9743 |
| 官方检查错误数 | 0 | 0 |

来源：[BM25 复跑](../records/bm25/review_bm25_20261103.json)、[混合检索](../records/hybrid/bge_hybrid_dev_20261103.json)。混合检索 p50/p95 为 5.9071/6.518 秒，输入 token 合计 6,252，输出为 0；每题 7 个 retrieval 事件，模型工具调用为 0。耗时包括每题进程、语料与索引加载及 API 查询。

混合检索比 BM25 多命中 2 题 top-10，Hit@1 减少 2 题。证据引用均通过原文跨度检查；答案采用抽取策略，AnswerEM 为 0。

[独立复跑](../records/hybrid/review_hybrid_20261104.json)的五项效果指标一致。49/50 题的文档顺序、引用、答案、状态全部一致；另 1 题顺序与引用变化，文档集合、答案和状态一致。

## 冻结 test50

冻结配置：[configs/final.json](../configs/final.json)。最终结果位于本地 `output/hybrid/test_base_final/`，包括 `results.jsonl`、`run_info.json`、`method_config.json`、50 份逐题 trace 和 `check_and_run.log`。

| 项目 | 结果 |
| --- | ---: |
| 完整覆盖 | 50 / 50 |
| ok | 50 |
| submission 检查错误数 | 0 |
| 平均 / p50 / p95 耗时（秒） | 6.0433 / 5.9056 / 7.0808 |
| 输入 / 输出 token | 6,100 / 0 |
| retrieval 事件 | 350 |

来源：[测试汇总](../records/hybrid/bge_test_final_20261104.json)、[冻结交付记录](../records/hybrid/final_delivery_20261104.json)。测试标签由教师持有，本地无 Hit 或答案正确率结果。

## 构建与资源

- 切块：size 1200、overlap 200、min_chars 120，共 321,659 个片段，全部通过原文逐字核对。
- BM25：k1=0.9、b=0.4、stopwords=true、capitalized_boost=1.25；842,329 个词项，29,159,319 条 postings，索引 207,921,947 bytes。索引元数据记录构建 74.039 秒。
- 向量：OpenRouter `baai/bge-m3`，1024 维，归一化后精确内积检索；索引 1,317,515,309 bytes。候选参数与 RRF k=60 见冻结配置。
- 索引编码成功响应累计输入 97,147,705 tokens。`encoding_seconds=40646.1825` 为批次请求耗时之和；`build_wall_seconds=4446.9358` 是最后一次续建调用的墙钟耗时。历次构建总墙钟时间和完整账单费用未统计。
- API 模型内部 revision 未固定。构建前段采用默认供应商路由，后段优先 DeepInfra，Parasail 备用。成功响应 token 不包含未知的失败请求计费。

## 验证与历史记录

课程测试 33 项、核心检索/证据测试 49 项、流水线/插件测试 14 项、向量后端测试 12 项，共 108 项通过。train 上已完成 dense-only 和 hybrid 各 3 题冒烟，完整指标以本页 dev50 为准。

早期 BM25 运行平均耗时约 5.98 秒，其后在当前环境复跑为 1.7549 秒；本页对照使用后者。早期类型化短跨度答案实验的 AnswerEM 同为 0，答案长度中位数 12.5 字符；配置保留在 `configs/bm25.span.json`。历史聚合保留在 `records/`，不覆盖原始数字。

ada-002 构建因余额不足停在 26,304 个片段，记录为 [ada_progress_20261103.json](../records/ada/ada_progress_20261103.json)。最终交付使用完整 BGE-M3 索引。
