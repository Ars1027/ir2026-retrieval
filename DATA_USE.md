# 数据来源与构建说明

数据来自 [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)。上游数据集卡标注为 MIT，许可文本保存在 `licenses/LICENSE_BrowseComp-Plus.txt`。文档来源 URL 保留在 `corpus/documents.jsonl` 中。

- 问题及标注：[Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus)，固定版本 `144cff8e35b5eaef7e526346aa60774a9deb941f`。
- 文档：[Tevatron/browsecomp-plus-corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)，固定版本 `b27b02bc3e45511b8b82a13e6f90ce761df726f6`。

构建时对原始问题 ID 与种子 2026 计算稳定哈希，排序后选取 200 道有效问题，再依次分配为训练 100、验证 50、测试 50。抽样没有使用任何模型的答题表现。三个划分的 ID 与归一化问题文本均不重叠。

保留选中题目全部 gold 与 evidence 文档，对每题以稳定哈希选取至多 30 篇上游难负例，再从剩余语料中按稳定哈希补足 10,000 篇。最终包含 1,185 篇必需文档、5,901 篇选取的难负例和 2,914 篇补充文档。正文不再切块或截断；不按问题建立文件夹，也不在文件名中标注相关性。

这份子集便于课程实验，但缩减语料会改变搜索难度，成绩不能与原始 100K 文档基准直接横比。原始基准公开，教师单独保管测试标签并不意味着题目从未被公开或被模型见过。

请遵循上游防污染说明，在课程内部使用解混淆后的问答和轨迹，不把这些内容上传到公开代码仓库或网页。项目包未包含其他研究者的已完成搜索轨迹。

引用文献：Zijian Chen et al. *BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent*. 2025. [arXiv:2508.06600](https://arxiv.org/abs/2508.06600)。
