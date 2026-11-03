# 成员 B 实现与验证记录（BM25 基础任务）

## 范围
仓库只包含一条固定方法：BM25 基线。向量空间模型不在此实现，只在 `retrieval/interfaces.py` 预留接口，由外部协作成员在附加题 1 中接入；接口默认关闭，基线执行路径不含任何向量检索。

## 代码
- `retrieval/interfaces.py`：`VectorRetriever` / `DocumentFusion` 抽象、`NullVectorRetriever` / `NullFusion` 默认实现、`validate_vector_hits` / `validate_ranked_docs` 输出校验、`"包.模块:工厂"` 后端加载、`interface_descriptor()` 机器可读规范。
- `pipeline/vector_stage.py`：仅在配置启用向量分支时执行；顺序为 BM25 → 文档聚合 → 插件检索（校验）→ 插件融合 → 共用 `assemble_result`；插件输出非法时记录 `vector_search.errors` 并降级回纯 BM25。
- `agent/solution.py`：默认入口；`vector.enabled=false` 时直接调用 `agent.bm25_solution.solve`，与基线执行路径和 trace 阶段完全一致。
- `pipeline/answer.py`：两种抽取式答案策略（默认句子级；`answer.span_selection=true` 时按问句类型挑实体跨度并把引文锚定到该跨度），另有 `model` 模式备用。
- `scripts/{run_baseline,summarize_runs}.py`、`run.sh`、`run.ps1`、`configs/{bm25,bm25.span,vector.example}.json`、`requirements/member_b.txt`（仅 numpy）、`VECTOR_PLUGIN_GUIDE.md`（协作者操作手册）。

## 验证
| 项 | 结果 |
| --- | --- |
| `python -m unittest discover -s tests` | 33 项通过（课程原始） |
| `python -m unittest discover -s tests/member_a` | 47 项通过（含答案策略） |
| `python -m unittest discover -s tests/member_b` | 13 项通过（接口规范、非法输出、插件加载失败、基线等价性、插件端到端） |
| dev 50（`output/member_b/dev_bm25_locked`） | Hit@1/5/10 = 0.20/0.30/0.40，EvidenceRecall@10 = 0.1999，AnswerEM = 0.00，completion = 1.00，平均 5.98 s/题 |
| test 50（`output/member_b/test_base_final_locked`） | 50/50 `ok`，`--submission` 检查 `valid: true, errors: []` |
| 单题 trace | `load_index → bm25_search → document_aggregate → read_evidence → answer`，全部为 `retrieval` 事件 |

## 交接给外部协作成员（附加题 1）
见 `VECTOR_PLUGIN_GUIDE.md`，要点：
1. 实现 `retrieval.interfaces.VectorRetriever` 子类，`search(query, top_k)` 返回 `[{"doc_id", "rank", "score"}]`。
2. 需要融合时实现 `DocumentFusion.fuse(bm25_docs, vector_docs, top_k) -> list[RankedDoc]`，并保留 `hits` 中真实的字符区间。
3. 复制 `configs/vector.example.json`，把 `retriever` / `fusion` 改成自己的 `"包.模块:工厂"`，`enabled` 置 true。
4. `python scripts/run_baseline.py --method final --config <自己的配置> --method-label bm25-v1+vector --split dev --output output/member_b/dev_vector --check --evaluate`。
5. 不要修改 `agent/bm25_solution.py`、`pipeline/result_builder.py`、`agent/solution.py` 的 BM25 分支与课程原始文件。
