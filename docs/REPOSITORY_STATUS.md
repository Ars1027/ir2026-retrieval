# Git 文件与本地产物

核对日期：2026-11-04。基础交付远程提交：`65a3ada2d2844c0026cea0ecaa948cbd2328ca91`。基础交付已核对远程文件内容；下表采用本次路径迁移后的名称，随本次提交推送，实验数字保持原值。

目录按检索方法和模块职责组织。

## 远程文件清单

| 文件 | origin/main |
| --- | --- |
| [`agent/__init__.py`](../agent/__init__.py) | 已推送 |
| [`agent/bm25_solution.py`](../agent/bm25_solution.py) | 已推送 |
| [`agent/solution.py`](../agent/solution.py) | 已推送 |
| [`check_results.py`](../check_results.py) | 已推送 |
| [`configs/bm25.json`](../configs/bm25.json) | 已推送 |
| [`configs/bm25.span.json`](../configs/bm25.span.json) | 已推送 |
| [`configs/default.json`](../configs/default.json) | 已推送 |
| [`configs/final.json`](../configs/final.json) | 已推送 |
| [`configs/model.example.json`](../configs/model.example.json) | 已推送 |
| [`configs/vector.ada.json`](../configs/vector.ada.json) | 已推送 |
| [`configs/vector.bge.dense.json`](../configs/vector.bge.dense.json) | 已推送 |
| [`configs/vector.bge.json`](../configs/vector.bge.json) | 已推送 |
| [`configs/vector.example.json`](../configs/vector.example.json) | 已推送 |
| [`evaluate.py`](../evaluate.py) | 已推送 |
| [`pipeline/__init__.py`](../pipeline/__init__.py) | 已推送 |
| [`pipeline/answer.py`](../pipeline/answer.py) | 已推送 |
| [`pipeline/evidence.py`](../pipeline/evidence.py) | 已推送 |
| [`pipeline/model_client.py`](../pipeline/model_client.py) | 已推送 |
| [`pipeline/result_builder.py`](../pipeline/result_builder.py) | 已推送 |
| [`pipeline/settings.py`](../pipeline/settings.py) | 已推送 |
| [`pipeline/vector_stage.py`](../pipeline/vector_stage.py) | 已推送 |
| [`records/bm25/environment.json`](../records/bm25/environment.json) | 已推送 |
| [`records/bm25/recheck_20261103.json`](../records/bm25/recheck_20261103.json) | 已推送 |
| [`records/hybrid/review_hybrid_20261104.json`](../records/hybrid/review_hybrid_20261104.json) | 已推送 |
| [`records/ada/ada_progress_20261103.json`](../records/ada/ada_progress_20261103.json) | 已推送 |
| [`records/hybrid/bge_hybrid_dev_20261103.json`](../records/hybrid/bge_hybrid_dev_20261103.json) | 已推送 |
| [`records/hybrid/bge_test_final_20261104.json`](../records/hybrid/bge_test_final_20261104.json) | 已推送 |
| [`records/bm25/decision.json`](../records/bm25/decision.json) | 已推送 |
| [`records/bm25/dev_bm25_metrics.json`](../records/bm25/dev_bm25_metrics.json) | 已推送 |
| [`records/bm25/baseline_environment.json`](../records/bm25/baseline_environment.json) | 已推送 |
| [`records/hybrid/final_delivery_20261104.json`](../records/hybrid/final_delivery_20261104.json) | 已推送 |
| [`records/bm25/review_bm25_20261103.json`](../records/bm25/review_bm25_20261103.json) | 已推送 |
| [`records/bm25/test_base_final_summary.json`](../records/bm25/test_base_final_summary.json) | 已推送 |
| [`requirements.txt`](../requirements.txt) | 已推送 |
| [`requirements/core.txt`](../requirements/core.txt) | 已推送 |
| [`requirements/pipeline.txt`](../requirements/pipeline.txt) | 已推送 |
| [`requirements/vector.txt`](../requirements/vector.txt) | 已推送 |
| [`retrieval/__init__.py`](../retrieval/__init__.py) | 已推送 |
| [`retrieval/bm25.py`](../retrieval/bm25.py) | 已推送 |
| [`retrieval/chunking.py`](../retrieval/chunking.py) | 已推送 |
| [`retrieval/contracts.py`](../retrieval/contracts.py) | 已推送 |
| [`retrieval/corpus_adapter.py`](../retrieval/corpus_adapter.py) | 已推送 |
| [`retrieval/document_rank.py`](../retrieval/document_rank.py) | 已推送 |
| [`retrieval/interfaces.py`](../retrieval/interfaces.py) | 已推送 |
| [`run.ps1`](../run.ps1) | 已推送 |
| [`run.py`](../run.py) | 已推送 |
| [`run.sh`](../run.sh) | 已推送 |
| [`scripts/build_bm25.py`](../scripts/build_bm25.py) | 已推送 |
| [`scripts/build_chunks.py`](../scripts/build_chunks.py) | 已推送 |
| [`scripts/build_vector.py`](../scripts/build_vector.py) | 已推送 |
| [`scripts/run_baseline.py`](../scripts/run_baseline.py) | 已推送 |
| [`scripts/summarize_runs.py`](../scripts/summarize_runs.py) | 已推送 |
| [`tests/core/fixtures.py`](../tests/core/fixtures.py) | 已推送 |
| [`tests/core/test_answer_strategies.py`](../tests/core/test_answer_strategies.py) | 已推送 |
| [`tests/core/test_bm25_rank.py`](../tests/core/test_bm25_rank.py) | 已推送 |
| [`tests/core/test_chunking.py`](../tests/core/test_chunking.py) | 已推送 |
| [`tests/core/test_contracts_settings.py`](../tests/core/test_contracts_settings.py) | 已推送 |
| [`tests/core/test_pipeline_assembly.py`](../tests/core/test_pipeline_assembly.py) | 已推送 |
| [`tests/pipeline/fixtures.py`](../tests/pipeline/fixtures.py) | 已推送 |
| [`tests/pipeline/test_vector_interface.py`](../tests/pipeline/test_vector_interface.py) | 已推送 |
| [`tests/test_method_protocol.py`](../tests/test_method_protocol.py) | 已推送 |
| [`tests/test_pipeline.py`](../tests/test_pipeline.py) | 已推送 |
| [`tests/vector_backend/test_backend.py`](../tests/vector_backend/test_backend.py) | 已推送 |
| [`vector_backend/__init__.py`](../vector_backend/__init__.py) | 已推送 |
| [`vector_backend/ada.py`](../vector_backend/ada.py) | 已推送 |
| [`verify_data.py`](../verify_data.py) | 已推送 |

## 本地交付与重建

| 内容 | 路径 | 获取方式 |
| --- | --- | --- |
| 课程语料与问题/标签 | `corpus/`、`datas/` | 课程渠道下载 |
| 切块与 BM25 索引 | `artifacts/chunks/`、`artifacts/bm25/` | README 构建命令 |
| BGE 索引与续建缓存 | `artifacts/bge_m3/` | API 编码重建或课程内部传递 |
| 完整 hybrid dev | `output/hybrid/bge_hybrid_dev_20261103/` | 当前工作区交接或重新运行 |
| 独立 hybrid 复跑 | `output/hybrid/review_hybrid_20261104/` | 当前工作区交接 |
| 最终 test50 | `output/hybrid/test_base_final/` | 当前工作区交接 |
| API 密钥 | 环境变量 `OPENROUTER_API_KEY` | 接手者自行配置 |

以上数据、索引、逐题结果与密钥由 `.gitignore` 排除；克隆仓库可获得代码和聚合记录，运行需要另外准备数据与索引。历史分工提示文件保留为本地文件，当前交接以 [HANDOFF.md](HANDOFF.md) 为准。
