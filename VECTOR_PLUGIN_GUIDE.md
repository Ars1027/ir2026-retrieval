# API 向量检索接入与验收

BM25 基线使用 `configs/bm25.json`。新增 `configs/vector.bge.json` 使用 `baai/bge-m3`，通过 `https://openrouter.ai/api/v1/embeddings` 编码全部原文片段。API 密钥只从 `OPENROUTER_API_KEY` 环境变量读取；禁止写入提交、配置快照或日志。

## 实现与接口

- `vector_backend/ada.py`：API 编码、1024 维归一化、FAISS IndexFlatIP、文档去重与 RRF。
- `scripts/build_vector.py`：读取成员 A 的 `chunks.jsonl` 和 `meta.json`，按批编码与恢复。
- `retrieval/interfaces.py`：`VectorRetriever.search(query, top_k)` 返回文档级排名；API 后端额外保留真实命中 `hits`。
- `DocumentFusion.fuse(bm25_docs, vector_docs, top_k)` 计算每路每文档一次 `1 / (rrf_k + rank)`，保留来源名次和原文跨度，最终最多 10 篇。
- `pipeline/vector_stage.py`：固定 retrieval 流程，复用公共证据/答案组装器。API 失败或非法返回报错，不把失败的混合检索包装成成功的 BM25 结果。

片段 `rank` 是该片段在向量命中中的真实名次，文档 `rank` 为去重后的连续名次，两者不能混淆。引用始终由原文跨度读取，不由嵌入模型生成。

## 构建与续建

安装 `requirements.txt` 和 `requirements/vector.txt`，在当前终端配置 `OPENROUTER_API_KEY`，然后运行：

```bash
python scripts/build_chunks.py
python scripts/build_bm25.py
python scripts/build_vector.py --config configs/vector.bge.json --workers 16
```

每批 64 片段，本轮命令最多 16 个在途请求（脚本默认 8 个），按原始片段顺序构建索引。`artifacts/member_b/bge_m3/batches/*.npz` 保存成功批次、输入摘要、API token 和耗时；中断后重跑同一命令复用批次，不重发已缓存请求。429/5xx 最多额外重试 3 次，遵循 `Retry-After`。其他 HTTP 错误立即失败。

完整成功后生成 `index.faiss`、`mapping.jsonl`、`metadata.json`。未完成索引不能加载。配置或源元数据不一致需要新目录；不覆盖旧实验。服务端模型别名无法保证内部不可变 revision，元数据会明确记录这一限制。

## 实验流程

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests/member_a -v
python -m unittest discover -s tests/member_b -v
python -m unittest discover -s tests/vector_backend -v
python scripts/run_baseline.py --method final --config configs/vector.bge.json --method-label bm25-bge-m3-rrf-v1 --split train --limit 3 --output output/member_b/bge_train --check --evaluate
python scripts/run_baseline.py --method final --config configs/vector.bge.json --method-label bm25-bge-m3-rrf-v1 --split dev --output output/member_b/bge_dev --check --evaluate
```

独立复制配置并设 `vector.dense_only = true` 可做 dense 排名诊断。应使用不同方法标签和输出目录；该诊断仍经过公共固定流水线，不能把整个查询耗时解释为纯向量搜索耗时。

完整 dev50 官方检查、评测和交叉复核完成后，决定最终方法并冻结配置 `frozen: true`，再执行：

```bash
python scripts/run_baseline.py --method final --config configs/vector.bge.json --method-label bm25-bge-m3-rrf-v1 --split test --output output/member_b/test_bge_final --check
```

未冻结时入口拒绝 test，test 不使用 `--limit`，不读取测试标签、不据测试反馈调参。成功返回 `ok` 和格式检查通过均不代表答案语义正确。

## 统计与历史结果

查询输入 token 来自 API `usage.prompt_tokens`，与答案模型输入合计；任一分量未知则总计为 `null`。索引构建 token 单独记录，不混入逐题用量。未知账单费用为 `null`，不以调用次数估价。原文、问题、引用、逐题结果和轨迹均保留本地；公开记录仅含聚合指标与配置/环境摘要。

README 与 `records/member_b` 中原有 BM25 Hit@10=0.40 等数据来自成员 A PR #1 的历史基线；原始产物未随 Git 合并传输。这些数字不能作为新 BGE-M3 hybrid 的结果。新方法是否完成、真实 dev 指标及最终输出以本轮新增记录和本地产物为准。

旧 `configs/vector.ada.json` 和 `artifacts/member_b/ada/` 为先前 ada-002 运行保留；BGE-M3 使用独立的 `artifacts/member_b/bge_m3/`。两者的模型、维度与 API 不同，缓存和索引不能混用。模块名 `vector_backend/ada.py` 为沿用的文件名，实际模型由配置决定。
