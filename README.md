# IR 2026：基础检索与附加题交接

基础阶段已完成：保真切块、BM25、BGE-M3 向量检索、文档级 RRF、原文引用、抽取式答案、统一运行与评测。默认方法为 **BM25 + BGE-M3 + RRF**，配置已冻结在 [configs/final.json](configs/final.json)。两道附加题在这套基线上继续开展：检索方法改进、小模型后训练。

- [实验结果与产物](docs/RESULTS.md)：完整 dev50 对照、test50 检查、资源统计。
- [附加题交接](docs/HANDOFF.md)：可复用模块、扩展入口、实验流程。
- [Git 文件核对](docs/REPOSITORY_STATUS.md)：远程已有文件与本地交付产物。
- [向量接口](VECTOR_PLUGIN_GUIDE.md)：编码、索引与融合契约。
- [课程协议](SPEC.md)与[数据使用约定](DATA_USE.md)。

## 当前结果

| 完整 dev50 | BM25 | BM25 + BGE-M3 + RRF |
| --- | ---: | ---: |
| Hit@1 | 0.20 | 0.16 |
| Hit@5 | 0.30 | 0.34 |
| Hit@10 | 0.40 | 0.44 |
| EvidenceRecall@10 | 0.199889 | 0.200857 |
| AnswerEM | 0.00 | 0.00 |
| 平均每题耗时（秒） | 1.7549 | 5.9743 |

两套 dev50 均通过官方检查。冻结方法的 test50 已完成，50 条结果均为 `ok`，提交检查错误数为 0。test 标签由教师持有，本地只报告覆盖、状态、引用和耗时。抽取式答案的 dev AnswerEM 为 0，答案质量是后续改进重点。

## 从代码重建

Python 环境与依赖见 `requirements.txt`；本次验证环境为 Python 3.13.12。将课程提供的 `corpus/`、`datas/` 放在项目根目录，在终端设置 `OPENROUTER_API_KEY`，然后运行：

```bash
python -m pip install -r requirements.txt
python verify_data.py --full
python scripts/build_chunks.py
python scripts/build_bm25.py
python scripts/build_vector.py --config configs/vector.bge.json --workers 8
```

全部 10,000 篇文档生成 321,659 个片段。向量模型为 OpenRouter `baai/bge-m3`，输出 1024 维归一化向量，FAISS 使用精确内积检索。批次缓存支持中断续建；完成后生成索引、ID 映射及元数据。

## 运行与验证

以下命令从项目根目录执行，输出目录使用新的名称。

```bash
# BM25 对照
python scripts/run_baseline.py --method bm25 --split dev --output output/bm25_dev_new --check --evaluate
# 冻结的混合方法
python scripts/run_baseline.py --method final --split dev --output output/hybrid_dev_new --check --evaluate
# dense-only 的 train 冒烟
python scripts/run_baseline.py --method final --config configs/vector.bge.dense.json --split train --limit 3 --output output/dense_train_new --check --evaluate
# 核对已有最终测试结果
python check_results.py --split test --run-dir output/member_b/test_base_final --submission
```

Windows 可使用 `./run.ps1`，其他平台可使用 `bash run.sh`，参数与统一入口相同。脚本通过 `IR2026_METHOD_CONFIG` 将配置传给课程运行器。

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests/member_a -v
python -m unittest discover -s tests/member_b -v
python -m unittest discover -s tests/vector_backend -v
```

最近验证共 108 项通过（33 + 49 + 14 + 12）。目录名沿用已有工程路径；交接以模块职责为准。
