# 成员 A 实现与验证记录

## 已实现
- `retrieval/contracts.py`：冻结 `Chunk` / `Hit` / `RankedDoc` 字段、构造与校验、切块配置解析、chunk_id 稳定命名。
- `retrieval/corpus_adapter.py`、`retrieval/chunking.py`：只读语料访问与保真切块（1200/200/120），`chunk.text == Corpus.text(doc_id)[start:end]`。
- `retrieval/bm25.py`、`retrieval/document_rank.py`：倒排索引（NumPy 并行数组，一次构建多次加载）、`BM25Retriever.search`、按最高排名片段聚合成文档。
- `retrieval/interfaces.py`：向量空间模型的预留接口（抽象类、默认空实现、输出校验、`包.模块:工厂` 装载），默认不启用。
- `pipeline/settings.py`、`evidence.py`、`answer.py`、`model_client.py`、`result_builder.py`：`IR2026_METHOD_CONFIG` 配置读取、逐字引用（支持把引文锚定到答案跨度）、extractive/model 两种答案策略、共用 `assemble_result`。
- `pipeline/vector_stage.py`：启用向量分支时的执行位置（BM25 → 聚合 → 插件检索 → 插件融合 → 组装）。
- `agent/bm25_solution.py`、`agent/solution.py`、`configs/{bm25,bm25.span,model.example,vector.example}.json`、`requirements/member_a.txt`。
- 构建脚本 `scripts/{build_chunks,build_bm25}.py`；统一运行入口 `scripts/run_baseline.py`（`--method bm25|final`）。

## 真实运行（本机）
| 阶段 | 命令 | 结果 |
| --- | --- | --- |
| 切块 | `python scripts/build_chunks.py` | 10,000 篇 → 321,659 片段，379.5 M 字符，414 MB，114.7 s |
| BM25 | `python scripts/build_bm25.py` | 29,159,319 postings，842,329 terms，208 MB，114.0 s |
| dev 50（句子答案） | `python scripts/run_baseline.py --method final --split dev --output output/member_b/dev_bm25_locked --check --evaluate` | 检查通过；GoldHit@1/5/10 = 0.20/0.30/0.40；EvidenceRecall@10 = 0.1999；AnswerEM = 0.00；completion = 1.00；平均 5.98 s/题 |
| dev 50（类型化短跨度） | 同上，`--config configs/bm25.span.json`，输出 `output/member_b/dev_bm25_span` | 检查通过；检索与证据指标不变；AnswerEM = 0.00；答案长度中位数 12.5 字符，且 50/50 落在自身引文中 |

## 答案策略说明
- 默认 `answer.span_selection = false`：从证据句子中挑与问题词/数字重合最多的一句，≤200 字符。
- 打开后按问句推断答案类型（标题/人名/机构/数量/日期），在 top-k 文档正文中生成候选跨度，用「句内关键词 + 类型 + 文档排名」打分，并把引文锚定到该跨度所在句子。
- 两种策略的 AnswerEM 都是 0.00：金标答案仅 8/50 出现在原引文中，20/50 出现在 top-10 文档正文，候选能产出 13/20，但文档中没有 `titled/named/引号` 之类线索，精确字符串无法可靠定位。提升 AnswerEM 需要 `answer.mode = "model"`（本机无密钥）。

## dev 侧对照（仅用于配置选择）
- 候选池 200 → 5000 片段：文档排名不变（0.40），说明瓶颈在排序而非候选数量。
- 600/100 切块重建后：Hit@10 0.38–0.40，与 1200/200 相当，故保留 1200/200 以降低索引成本。
- 上述试验产物已删除，结论记录在此。

## 接口交接（给成员 B 与外部协作者）
```python
from retrieval.bm25 import BM25Retriever          # BM25Retriever(index_dir, settings).search(query, top_k) -> list[Hit]
from retrieval.document_rank import aggregate_documents   # (hits, top_k) -> list[RankedDoc]
from pipeline.settings import load_settings        # IR2026_METHOD_CONFIG 或 configs/bm25.json
from pipeline.result_builder import assemble_result  # (question, ranked_docs, tools, settings) -> result dict
from retrieval.interfaces import VectorRetriever, DocumentFusion   # 附加题 1 的接入点
```
构建命令：`python scripts/build_chunks.py`、`python scripts/build_bm25.py`；产物路径 `artifacts/member_a/chunks`（`chunks.jsonl` + `meta.json`）、`artifacts/member_a/bm25`。

## 未完成/阻塞
- 未使用真实问答模型：`configs/model.example.json` 只提供配置样例，本机没有密钥，答案策略为 extractive，token 记 0。
- 逐题答案语义正确性未评估（AnswerEM 仅作辅助，当前为 0）。
- 最终 test 集由成员 B 冻结配置后运行；成员 A 只读复核。
