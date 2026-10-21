# 100+100 检索评测记录

这份记录对应 2026-10-14 在本地设备上的一次可复现实验。数据准备使用固定随机种子 `42`：CMRC2018 开发集抽取 100 道问题；HotpotQA distractor validation 前 100 条样本抽取 100 道问题。两者的参考答案和相关文档标签未写入检索索引。

## 评测协议

- Chunk：character，大小 700，重叠 100。
- 候选：先召回 30 个 Chunk，再合并同源 Chunk，保留前 5 个不同文档。
- 方法：BM25、BGE-M3 Dense、Hybrid（Dense + BM25 + Qdrant RRF）、Hybrid + Cross-Encoder。
- 指标：Recall@5、Precision@5、MRR@5、NDCG@5。
- 评测单位：来源文档。HotpotQA 的相关文档来自支持事实标注。
- BGE-M3 使用 CUDA；`BAAI/bge-reranker-base` 使用 CPU，避免 4GB 显存同时加载两个模型。

## 结果

| 数据集 | 方法 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | 检索 ms | 重排 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| CMRC2018 | BM25 | 1.0000 | 0.2000 | 0.9750 | 0.9815 | 109.5 | 0 |
| CMRC2018 | Dense | 0.9900 | 0.1980 | 0.9608 | 0.9682 | 44.1 | 0 |
| CMRC2018 | Hybrid | 1.0000 | 0.2000 | 0.9867 | 0.9900 | 165.4 | 0 |
| CMRC2018 | Hybrid + Rerank | 1.0000 | 0.2000 | 0.9950 | 0.9963 | 305.9 | 7620.7 |
| HotpotQA | BM25 | 0.7350 | 0.2940 | 0.8675 | 0.7112 | 16.0 | 0 |
| HotpotQA | Dense | 0.9050 | 0.3620 | 0.9425 | 0.8655 | 42.4 | 0 |
| HotpotQA | Hybrid | 0.8700 | 0.3480 | 0.9292 | 0.8297 | 59.0 | 0 |
| HotpotQA | Hybrid + Rerank | 0.9350 | 0.3740 | 0.9670 | 0.9224 | 200.3 | 3438.0 |

## 运行环境

- GPU：NVIDIA GeForce RTX 3050 Laptop，4GB。
- CPU：Intel i5-11400H，6 核 12 线程；内存 16GB。
- Python 3.12.10；torch 2.14.1+cu126；FlagEmbedding 1.4.2；transformers 4.57.6；ranx 0.3.21；qdrant-client 1.19.1。
- 原始 JSON、运行明细和完整 JSON 汇总位于 `.test-tmp/retrieval-eval-100/`，该目录被 Git 忽略。

这是固定子集上的项目评测结果，不是官方全量排行榜成绩。它评测的是文档证据召回，不包含 LLM 最终答案生成准确率。
