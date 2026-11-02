# 成员 B 实现与验证记录（2026-11-02）

状态：独立实现完成，真实端到端受成员 A 接口和模型资源阻塞；未冻结、未运行 test。未修改课程代码、成员 A 文件或默认 agent/solution.py。

## 修改文件
retrieval/{embedding,dense,fusion}.py；pipeline/hybrid.py；agent/{dense_solution,hybrid_solution}.py；configs/{dense,hybrid}.json；requirements/member_b.txt；requirements.txt；scripts/{build_dense,run_baseline,summarize_runs}.py；run.sh；run.ps1；README.md；tests/member_b/；本记录。

## 验证
- `python verify_data.py --full`：通过，10000 文档、200 问题及完整摘要。
- `python -m unittest discover -s tests -v`：课程原始 33 项通过。
- `python -m unittest discover -s tests/member_b -v`：12 项通过。涵盖 FAISS 中文路径、映射重载、维度/版本/源元数据不匹配、空候选、RRF 手算/重复/同分、流水线日志、失败传播、配置传递与聚合隐私。
- 索引测试使用虚构 encoder，不代表真实模型语义效果。

## 环境及模型
基准 commit：973da053caa92103c319c97786ed9665b9aaead7。
Windows；物理内存约 16.87 GB；numpy 2.4.4、faiss-cpu 1.13.0。已安装 FAISS，torch/sentence-transformers 与本地模型权重缺失。
配置 Qwen/Qwen3-Embedding-0.6B，revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3，1024 维、8192 token、CPU batch 1。模型卡：https://huggingface.co/Qwen/Qwen3-Embedding-0.6B 。查询加指令、文档无指令、向量归一化；超长片段报错，禁止截断。中文路径通过字节序列化读写 FAISS。

## 交接阻塞与未运行项
缺少 A 的 retrieval.document_rank.aggregate_documents、retrieval.bm25.BM25Retriever、pipeline.settings.load_settings、pipeline.result_builder.assemble_result、requirements/member_a.txt，以及切块/BM25产物。构建入口暂按 chunks.jsonl + metadata.json 读取，必须与 A 交付格式核对；生产层未复制 A 实现。
真实 Qwen 语义验证、全量向量构建、train dense/hybrid 冒烟、hybrid dev50、BM25只读复跑及交叉复核均未执行。Hit@1/5/10、EvidenceRecall@10、AnswerEM、真实成本与加载/查询时间均为 null（尚未运行），没有真实结果产物。Windows spawn 模型开销尚未测量。
因此不满足配置冻结和 test50 运行条件，默认入口保持原模板。依赖到齐后按 README 构建运行，使用独立新输出目录；完整 dev 执行官方 --submission 检查和 evaluate，再聚合指标。不得将本记录的单元测试视为端到端完成。
