# 成员 B 实现与验证记录

本文件记录当前基础交付状态；早前一次仅完成独立模块与虚构 fixture 的记录已被本次实现取代。

## 已实现
- `retrieval/embedding.py`：编码器接口，`hf`（Qwen3-Embedding-0.6B，模型卡查询指令、末位池化、L2 归一化）与 `hashing`（离线确定性回退）两种后端，`auto` 记录回退原因。
- `retrieval/dense.py`：全量切块向量化、`faiss.IndexFlatIP`（无 FAISS 时自动退化为 NumPy 精确内积）、维度/切块指纹/语料摘要校验。
- `retrieval/fusion.py`：文档级 RRF，`sum(1/(k+rank))`，每路每文档只计一次，保留来源排名与真实片段跨度，同分按 `doc_id` 稳定。
- `pipeline/hybrid.py`：固定 hybrid 流水线（两路召回 → 各自文档聚合 → RRF → 共用 `assemble_result`）。
- `agent/{dense,hybrid}_solution.py`、冻结默认入口 `agent/solution.py`、`scripts/{build_dense,run_baseline,summarize_runs}.py`、`run.sh`、`run.ps1`、两份配置与依赖入口。

## 实际运行（本机）
| 运行 | 目录 | 结果 |
| --- | --- | --- |
| dense 冒烟 train 5 | `output/member_b/train_dense_smoke` | 完成，检查通过（子集评测需 `--allow-subset`） |
| hybrid 冒烟 train 5 | `output/member_b/train_hybrid_smoke` | 完成 |
| hybrid dev 50 | `output/member_b/dev_hybrid` | 官方 `--submission` 检查通过，GoldHit@10 = 0.20 |
| BM25 dev 50（只读复跑） | `output/member_b/review_bm25` | 检查通过，GoldHit@10 = 0.40 |
| 最终 test 50 | `output/member_b/test_base_final` | 官方 `--submission` 检查通过（无标签，不计算 Hit） |

聚合指标见 `records/member_b/*.json`；原始结果、trace 与索引只在本地。

## 资源阻塞（如实记录）
- 本机无法访问 huggingface.co（ConnectTimeout），计划中的 `Qwen/Qwen3-Embedding-0.6B` 无法下载，因此 dense 分支使用 `hashing` 回退后端：文档与 meta 均标注 `"model": "hashed-features"` 与回退原因，**不是语义编码器**。
- 纯 CPU（12 核，无 CUDA）。全量 321,659 个切块用 Transformer 编码在本机不可行；hashing 回退耗时 240 s，索引 668 MB。
- 因此真实模型的 token 成本、语义召回与 GPU 构建成本均为 `null`，未编造。

## 冻结决定
RRF 实现正确且可复现（成员 A 只读复跑得到相同指标），但缺少语义编码器时 dense 分支噪声大，融合后 GoldHit@10 由 0.40 降到 0.20。最终基础交付冻结为 BM25：`agent/solution.py` → `agent/bm25_solution.py`，配置 `configs/bm25.json`。拿到可用的嵌入模型后重新执行 `python scripts/build_dense.py --backend hf` 并复跑 hybrid 命令即可重新评估。
