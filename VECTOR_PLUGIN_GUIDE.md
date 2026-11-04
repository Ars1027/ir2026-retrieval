# 向量检索与融合接口

当前后端是 `vector_backend/ada.py` 中的 API 编码器、FAISS 精确内积检索器和 RRF。默认冻结配置为 `configs/final.json`；开发配置为 `configs/vector.bge.json`，模型为 `baai/bge-m3`（1024 维）。密钥读取 `OPENROUTER_API_KEY`。

## 数据与排名

`VectorRetriever.search(query, top_k)` 返回文档级列表，每项包含 `doc_id`、从 1 开始的 `rank`、有限 `score`。当前后端还返回 `hits`，保存命中片段的 `chunk_id`、`doc_id`、`start`、`end`、`score`、`rank`、`source`。片段 rank 表示原始片段排名，外层 rank 表示去重后的文档排名。

`DocumentFusion.fuse(bm25_docs, vector_docs, top_k)` 返回 `RankedDoc` 列表，包含 `doc_id`、`rank`、`score`、`hits`。RRF 按各路文档排名求和 `1 / (rrf_k + rank)`，每路同文档贡献一次，保留两路命中跨度和来源排名；同分按文档 ID 排序。

新后端通过配置的 `vector.retriever`、`vector.fusion` 指定 `module:factory`。工厂接收完整 settings，分别返回 `VectorRetriever`、`DocumentFusion` 实例。公共接口在 `retrieval/interfaces.py`。

## 构建与恢复

```bash
python scripts/build_vector.py --config configs/vector.bge.json --workers 8
```

脚本读取 `chunks.jsonl` 与 `meta.json`，每 64 个片段保存一份缓存。429 大批请求拆成最多 8 条文本的子批；瞬时错误有限重试。工作线程并发请求，按原顺序写索引和映射。

完成产物为 `index.faiss`、`mapping.jsonl`、`metadata.json`，缓存位于 `batches/`。重启命令复用成功批次。更换模型或切块配置时使用新的索引目录。源摘要、模型参数和完整标记在加载时校验。

向量先归一化，再使用 `IndexFlatIP`。API 返回维度与输入数量均经过校验。查询成本计入每题 usage，索引编码成本单独统计。API 服务端模型采用别名，内部 revision 未固定；独立复跑出现过一题顺序/引用差异，聚合指标一致。

## 流水线接入

`pipeline/vector_stage.py` 调用检索和融合，再复用 `assemble_result`。`vector.dense_only=true` 跳过 BM25，配置见 `configs/vector.bge.dense.json`。API 或向量校验失败返回错误状态；实际步骤写入 retrieval 日志。

运行命令、正式结果和附加题验收分别见 [README](README.md)、[实验结果](docs/RESULTS.md)、[交接说明](docs/HANDOFF.md)。
