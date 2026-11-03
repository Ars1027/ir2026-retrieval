# 附加题 1 接入指南：向量空间模型（Vector Plugin Guide）

面向对象：负责附加题 1（向量空间模型）的 GitHub 协作成员。
适用版本：`vector-interface.v1`（`retrieval/interfaces.py`）。

本仓库当前只保留**一条**方法：纯粹的 BM25 基线。历史上那套向量检索与融合实现（含相关模块、配置、脚本、测试）已**全部删除**，不需要阅读、不需要兼容、也没有旧接口需要迁移。你要做的只有一件事：在预留接口处实现一个后端，然后跑同一套评测脚本给出可比数字。

---

## 1. 你的任务与边界

**必须实现**
1. 一个 `VectorRetriever` 子类：输入 `query` 与 `top_k`，输出文档级的 `doc_id` 排名与 `score`。
2. 一个 `DocumentFusion` 子类（可选但推荐）：决定最终 10 篇文档的顺序，或直接沿用 BM25 的排序。

**你可以新增**
- `vector_backend/`（你自己的包：索引构建、编码、检索、融合）
- `scripts/build_vector.py`（你的索引构建脚本）
- `configs/vector.<你的名字>.json`（你的方法配置）
- `tests/vector_backend/`（你的模块测试）
- `records/vector/<你的名字>_*.json`（你的运行记录）
- `requirements/vector.txt`（你的额外依赖，如 `torch` / `transformers` / `faiss-cpu`）

**你不可以改**
- `agent/bm25_solution.py`、`pipeline/result_builder.py`、`pipeline/evidence.py`、`pipeline/answer.py`
- `agent/solution.py` 的 BM25 分支（只允许通过配置启用/关闭向量分支，不改代码）
- `retrieval/interfaces.py` 的接口签名（要扩展先开 issue，不要直接改）
- 课程原始文件：`run.py`、`check_results.py`、`evaluate.py`、`ir2026.py`、`tools/`、`examples/`、`SPEC.md` 等
- `corpus/`、`datas/` 下任何文件，以及 `records/member_{a,b}/` 里已有的记录

边界原则：基线代码的任何字节都不应因为你的改动而受影响。合入前会检查 `python -m unittest discover -s tests/member_b` 仍然全绿，并且用 `configs/bm25.json` 跑出的 dev 指标必须仍是下面这份数字。

---

## 2. 当前基线：方法与实测结果

基线是单条固定流水线，每题一次 `retrieval` 事件，无模型自主工具调用、无 LLM 生成。

**方法链**：保真切块（原文 `[start:end]`）→ BM25 倒排检索 → 片段按最优 `chunk_id` 聚合为文档排名 → 逐字原文引用 → 抽取式短答案。

| 项 | 取值 |
|---|---|
| 切块 | size 1200 / overlap 200 / min_chars 120 → 10,000 篇 → **321,659 片段** |
| BM25 | `k1=0.9`，`b=0.4`，`use_stopwords=true`，`capitalized_boost=1.25` |
| 索引规模 | 842,329 terms，29.16 M postings，208 MB，构建 113.96 s |
| 候选池 | `candidate_k = 200` 片段 |
| 文档排名 | 以该文档最优片段的名次/得分排序，取前 10 |
| 证据 | 对前 10 篇文档各取其最优片段，裁剪到 20–1200 字符、去重后最多 5 段，必须逐字等于 `Corpus.text(doc_id)[start:end]` |
| 答案 | `extractive`：在证据句子中挑与问题词/数字重合最多的一句，≤200 字符；不使用模型，token 记 0 |

**dev 50 实测（官方 `evaluate.py`，`output/member_b/dev_bm25_locked`）**

| 指标 | 数值 |
|---|---|
| GoldHit@1 | **0.20**（10/50 题） |
| GoldHit@5 | **0.30**（15/50 题） |
| GoldHit@10 | **0.40**（20/50 题） ← 主指标 |
| EvidenceRecall@10 | **0.1999** |
| AnswerEM | 0.00 |
| completion_rate | 1.00 |
| citation_span_match_rate | 1.00 |
| 平均耗时 | 5.98 s/题（其中约 5.6 s 是课程运行器的语料初始化，检索本身 `bm25_search` 仅 32.6 ms） |
| 单题日志阶段 | `load_index → bm25_search → document_aggregate → read_evidence → answer`，共 5 条 |

test 50 已用同一冻结配置跑过并通过 `check_results.py --submission`（`valid: true, errors: []`，`status: all ok`，平均 10 篇文档 / 5 段引用）。test 没有标签，本地不算 Hit，**只允许在方法冻结后跑一次**。

**你的目标**：在 dev 50 上 `GoldHit@10` 超过 0.40，且 `EvidenceRecall@10` 不低于 0.1999、`completion_rate` 保持 1.00、引用仍逐字有效。

---

## 3. 接口规范

定义在 `retrieval/interfaces.py`，也可直接打印机器可读版本：

```powershell
python -c "from retrieval import interfaces; import json; print(json.dumps(interfaces.interface_descriptor(), ensure_ascii=False, indent=2))"
```

### 3.1 `VectorRetriever`（必须实现）

```python
class VectorRetriever(ABC):
    name: str                                        # 写进日志的短标识
    def search(self, query: str, top_k: int) -> list[dict]: ...   # 抽象方法
    def describe(self) -> dict: ...                  # 非敏感元数据，写入 trace
    def close(self) -> None: ...                     # 释放资源
```

输入：`query` 为题目原文（非空 `str`）；`top_k` 为正整数（来自配置 `vector.top_k`）。
输出：**按相关度降序**的列表，每项：

```python
{"doc_id": "课程原始 doc_id", "rank": 1, "score": 0.823}
```

硬性约束（`interfaces.validate_vector_hits` 会逐条校验，违反即视为失败）：
- `len(hits) <= top_k`
- `rank` 必须等于该项在列表中的位置，从 1 开始连续，不跳号
- `doc_id` 必须是课程语料里的原始字符串，且列表内不重复
- `score` 必须是有限浮点数（`nan` / `inf` 拒绝），越大越相关
- 分数只在你自己这个后端里可比，不要试图与 BM25 的分数运算

`close()` 会在进程退出前调用；`describe()` 的返回值会原样写进 `load_indexes` 事件的 `output.vector`，请只放非敏感信息（后端名、模型名、revision、向量维度、索引条目数）。

### 3.2 `DocumentFusion`（可选，推荐）

```python
class DocumentFusion(ABC):
    name: str
    def fuse(self, bm25_docs: list, vector_docs: list, top_k: int) -> list[dict]: ...
```

- `bm25_docs`：`RankedDoc` 列表（最多 100 条，已按 BM25 排序）。结构见 `retrieval/contracts.py`：
  `{"doc_id": str, "rank": int, "score": float, "hits": [Hit, ...]}`
- `vector_docs`：你的 `search` 输出（已通过校验）
- 返回：最多 `top_k` 个 `RankedDoc`，`rank` 从 1 连续、`doc_id` 不重复（由 `interfaces.validate_ranked_docs` 校验）

**关键约束**：`hits` 里必须带着真实的字符区间，证据阶段要用它切原文。最省事的做法是直接复用 `bm25_docs` 里的 dict，只改顺序和 `rank`；如果只想保留分数、丢掉 `hits`，运行会因取不到原文引用而产出 `no_answer`。

### 3.3 装载方式

配置里的字符串形如 `"包名.模块:工厂函数"`，工厂函数接收完整的 settings 字典：

```python
def create_retriever(settings: dict) -> VectorRetriever: ...
def create_fusion(settings: dict) -> DocumentFusion: ...
```

导入失败、工厂抛错、返回类型不对，都会抛 `retrieval.interfaces.RetrievalBackendError`，信息里带具体模块名与原因。

### 3.4 执行位置与容错

`pipeline/vector_stage.py` 是唯一调用你的地方，顺序固定：

```
load_indexes → bm25_search → document_aggregate → vector_search → fuse → read_evidence → answer
```

- 你的输出先校验，再融合。校验失败时：`vector_search` 事件的 `output.errors` 会记录问题列表，本轮**降级为纯 BM25 排名**继续跑完，不会中断 50 题。
- 你的 `search` 抛异常则整题失败，因此请在内部处理网络/显存等异常并返回空列表或降级结果。
- 没有配置 `fusion` 时使用 `NullFusion`（原样保留 BM25 排序），此时向量分支只用于“记录与实验对照”。

---

## 4. 接入步骤（照着做即可）

### 步骤 1：安装依赖

```powershell
python -m pip install -r requirements.txt -r requirements/vector.txt
```

`requirements/vector.txt` 由你新建，基线不需要这些包。

### 步骤 2：新建你的后端包

```text
vector_backend/
    __init__.py
    backend.py      # create_retriever / create_fusion 及实现类
    index.py        # 向量索引的构建与加载
scripts/build_vector.py     # 构建入口（参数化，路径不写死）
```

`vector_backend/backend.py` 最小模板：

```python
"""Vector-space backend for extension point 1."""
from retrieval import interfaces


class MyVectorRetriever(interfaces.VectorRetriever):
    name = "my-vector-v1"

    def __init__(self, settings):
        self.index_dir = settings["vector_index_dir"]
        self.model_name = settings["vector"]["model"]      # 你自己的配置键
        self._index = load_index(self.index_dir)            # 你的实现
        self._encoder = load_encoder(self.model_name)       # 你的实现

    def search(self, query, top_k):
        vector = self._encoder.encode_query(query)
        hits = self._index.search(vector, top_k)            # [(doc_id, score), ...]
        return [{"doc_id": doc_id, "rank": rank, "score": float(score)}
                for rank, (doc_id, score) in enumerate(hits, 1)][:top_k]

    def describe(self):
        return {"name": self.name, "model": self.model_name,
                "index_dir": str(self.index_dir), "items": len(self._index)}

    def close(self):
        self._index = None


class RankFusion(interfaces.DocumentFusion):
    """示例：把向量排名前 n 篇提到最前，其余保持 BM25 顺序。"""
    name = "vector-first"

    def __init__(self, promote=3):
        self.promote = promote

    def fuse(self, bm25_docs, vector_docs, top_k):
        by_id = {doc["doc_id"]: doc for doc in bm25_docs}
        ordered, seen = [], set()
        for hit in vector_docs[: self.promote]:
            doc = by_id.get(hit["doc_id"])
            if doc is not None:
                ordered.append(doc)
                seen.add(doc["doc_id"])
        ordered += [doc for doc in bm25_docs if doc["doc_id"] not in seen]
        from retrieval import contracts
        return [contracts.make_ranked_doc(doc["doc_id"], rank, doc["score"], doc["hits"])
                for rank, doc in enumerate(ordered[:top_k], 1)]


def create_retriever(settings):
    return MyVectorRetriever(settings)


def create_fusion(settings):
    return RankFusion(promote=int(settings["vector"].get("promote") or 3))
```

### 步骤 3：构建索引（你自己的脚本）

要求：
- 只读 `artifacts/member_a/chunks/chunks.jsonl`（字段 `chunk_id/doc_id/start/end/text`）与 `corpus/`；
- 输出到 `artifacts/vector/<你的名字>/`，附 `meta.json`（编码器名、revision、维度、条目数、构建秒数），便于记录复现；
- 产物在 `.gitignore` 覆盖范围内，**不要提交**。

```powershell
python scripts/build_vector.py --out-dir artifacts/vector/my-vector --model <你的模型>
```

### 步骤 4：写配置

复制模板并改两处 `retriever` / `fusion`：

```powershell
Copy-Item configs/vector.example.json configs/vector.myname.json
```

```json
{
  "method": "bm25-v1+vector",
  "vector_index_dir": "artifacts/vector/my-vector",
  "vector": {
    "enabled": true,
    "retriever": "vector_backend.backend:create_retriever",
    "fusion": "vector_backend.backend:create_fusion",
    "top_k": 100,
    "model": "你的模型名",
    "index_dir": "artifacts/vector/my-vector"
  }
}
```

`configs/vector.myname.json` 只需要写你要覆盖的键；其余键会从默认值继承（`candidate_k=200`、`bm25` 参数、`retrieval.final_docs=10`、`evidence_docs=5` 等请保持基线值，否则数字无法与基线比较）。

### 步骤 5：先粗测再全测

```powershell
# 冒烟：10 题，确认链路通
python scripts/run_baseline.py --method final --config configs/vector.myname.json `
    --method-label bm25-v1+vector --split dev --limit 10 `
    --output output/vector/smoke_10 --check

# 正式：dev 50 题 + 评测
python scripts/run_baseline.py --method final --config configs/vector.myname.json `
    --method-label bm25-v1+vector --split dev `
    --output output/vector/dev_my_vector --check --evaluate
```

输出目录必须为空（脚本不会覆盖历史运行）；每次运行会把配置快照写成 `output/.../method_config.json`，汇报时请一并发出来。

### 步骤 6：生成可提交的记录

```powershell
python scripts/summarize_runs.py --run-dir output/vector/dev_my_vector --split dev `
    --out records/vector/my_vector_dev_metrics.json
```

`records/` 只放聚合数字，禁止出现题面、gold 文档、答案原文。

---

## 5. 自检清单

```powershell
# 1) 接口契约自检（返回空列表说明输出结构合法）
python -c "import json;from retrieval import interfaces;from pipeline.settings import load_settings;s=load_settings('configs/vector.myname.json');r=interfaces.build_vector_retriever(s);hits=r.search('Halden mill wheel', 10);print(interfaces.validate_vector_hits(hits, 10));print([h['doc_id'] for h in hits][:5])"

# 2) 基线未被破坏（必须 13 项全过）
python -m unittest discover -s tests/member_b -v

# 3) 课程与成员A测试必须仍全绿
python -m unittest discover -s tests
python -m unittest discover -s tests/member_a

# 4) 单题 trace 校验：应看到 vector_search / fuse 事件，且 errors 为空
python -c "import json,glob;f=sorted(glob.glob('output/vector/dev_my_vector/traces/*.jsonl'))[0];print([ (e['seq'],e['tool']) for e in map(json.loads,open(f,encoding='utf-8'))]);"
```

常见失败对照：

| 现象 | 原因 | 处理 |
|---|---|---|
| `cannot import 'xxx' module` | `retriever` 里写的包名/模块名不对，或不在仓库根目录 | 从仓库根目录启动，写成 `包名.模块:工厂` |
| `did not return a VectorRetriever instance` | 工厂返回了别的对象，或类没继承抽象基类 | 继承 `interfaces.VectorRetriever` 并实现 `search` |
| `rank must equal position` | 排序后重新编号时跳号/倒序 | `enumerate(hits, 1)` 连续编号 |
| `duplicate doc_id` | 同一文档多个片段被当成多条文档 | 先按 `doc_id` 聚合，再做文档级排序 |
| `score must be finite` | 相似度溢出或除零 | 归一化后再输出 |
| 结果全是 `no_answer` | 你的 `RankedDoc` 丢了 `hits`，证据阶段取不到原文 | 复用 `bm25_docs` 的 dict，别重建空 `hits` |
| `vector_search` 事件 `errors` 非空 | 输出结构违规，本轮已降级为 BM25 | 按 `errors` 逐条修，直到为空 |

---

## 6. 评测协议（必须遵守）

1. **只用 dev 调参**。`datas/dev.jsonl` + `datas/dev_labels.jsonl` 是唯一允许反复使用的带标签集合；参数、prompt、模型选择都在这里定。
2. **test 只在冻结后跑一次**，且必须过官方检查：

   ```powershell
   python scripts/run_baseline.py --method final --config configs/vector.myname.json `
       --method-label bm25-v1+vector --split test --output output/vector/test_my_vector --check
   ```
3. **不得**把标签、gold 文档写进代码、配置、日志或提交历史；不得把 `datas/`、`corpus/`、`artifacts/`、`output/` 提交。
4. 汇报必须包含：`run_info.json` 内容、`metrics.json` 五项（Hit@1/@5/@10、EvidenceRecall@10、AnswerEM）、`completion_rate`、平均耗时、以及相对基线 0.40 的变化。
5. 主指标 `Hit@10` 与他人比较时，必须同 split、同 `final_docs=10`、同评测脚本；不允许用 chunk 命中冒充文档命中。

**验收线**：`GoldHit@10 > 0.40` 且 `EvidenceRecall@10 >= 0.1999`、`completion_rate == 1.00`、`citation_span_match_rate == 1.00`、`tests/member_b` 全绿、`configs/bm25.json` 跑出的 dev 数字仍是 0.20 / 0.30 / 0.40。

---

## 7. 建议的方法路线（基于本基线）

基线在文档级 Hit@10 只有 0.40，说明瓶颈主要在“文档粒度排序”。三条路线按性价比排序：

**路线 A：BM25 召回池 + 语义重排（推荐先做）**
`top_k` 设 100–200，用 BM25 的 `bm25_docs` 当候选池，对候选文档（或其特征片段）编码，按语义相似度重排前 10。`fusion` 直接返回重排结果，`hits` 复用 BM25 的。改动量最小，且不会把 BM25 已命中的文档挤掉。

**路线 B：文档级 RRF 融合**
两路各自给文档排名，`score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vector(d))`，`k` 常取 60。实现要点：
- 只对 `bm25_docs` 中已存在的 `doc_id` 计算（这些文档自带有效 `hits`，证据阶段能取原文）；
- 若向量路命中了 BM25 候选池之外的文档，需要自己从 `chunks.jsonl` 补齐该文档的片段与区间，否则证据阶段无法引用；
- `k`、`top_k`、是否加权（如 `0.7 / 0.3`）全部在 dev 上定，不要把 dev 数字反复当 test 用。

**路线 C：纯向量检索**
`search` 直接返回向量 top-k 文档，`fusion` 用 `NullFusion` 或简单合并。可行但风险最大：长文档的整篇向量容易稀释细节，本基线场景下通常弱于路线 A/B。若要尝试，请先只跑 dev 10 题筛掉明显退化的配置。

工程注意：
- 编码器选中文/英文混合能力强的模型，向量归一化后用内积等价余弦；
- 若无法访问 Hugging Face，请把模型权重放在本地目录并用绝对路径加载，同时在 `describe()` 与 `meta.json` 里如实写明模型与 revision；
- 编码 32 万片段在纯 CPU 上不可行，建议按文档聚合后编码（10,000 篇）或只在查询时对候选池编码；
- 保持确定性：固定 `torch` 线程数与随机种子，否则同配置两次 dev 结果会漂移。

---

## 8. Git 与协作规范

1. 从 `main` 拉分支：`git checkout -b vector/<你的名字>`。
2. 提交只包含：你的包、脚本、配置、测试、记录、`requirements/vector.txt`、必要的文档更新。
3. 不要改 `retrieval/interfaces.py`、`pipeline/vector_stage.py`、`agent/solution.py`；确需扩展时先提 issue 说明理由，由仓库负责人改。
4. PR 描述模板：

   ```text
   方法：<一句话>
   模型：<名称 + revision>
   构建：python scripts/build_vector.py --out-dir ... （耗时 X s，索引 Y MB）
   dev：命令 + Hit@1/@5/@10 + EvidenceRecall@10 + completion + 平均耗时
   test：命令 + check_results 输出（是否 valid）
   与基线差异：Hit@10 0.40 -> ?
   已知问题：<如实写>
   ```
5. 合入前自检：`tests` / `tests/member_a` / `tests/member_b` 三套全绿，且基线 dev 指标未变。

---

## 9. 一页速查

| 事项 | 值 |
|---|---|
| 接口版本 | `vector-interface.v1` |
| 接口模块 | `retrieval/interfaces.py` |
| 执行位置 | `pipeline/vector_stage.py` |
| 默认配置 | `configs/bm25.json`（`vector.enabled = false`） |
| 模板配置 | `configs/vector.example.json` |
| 统一入口 | `scripts/run_baseline.py --method final --config <cfg>` |
| 运行器适配器 | `agent/solution.py` → 关闭时直接走 `agent.bm25_solution` |
| 基线 dev 主指标 | GoldHit@10 = **0.40**，EvidenceRecall@10 = 0.1999 |
| 基线 dev 运行 | `output/member_b/dev_bm25_locked` |
| 基线 test 运行 | `output/member_b/test_base_final_locked`（`valid: true`） |
| 输出契约 | `[{doc_id, rank, score}]`，rank 1..n，doc_id 唯一，score 有限，len ≤ top_k |
| 融合契约 | `list[RankedDoc]`，≤ top_k 个唯一文档，`hits` 必须带真实区间 |
| 记录目录 | `records/vector/<你的名字>_*.json` |
