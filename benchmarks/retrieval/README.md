# 中英文检索评测入门

中文使用 [CMRC2018](https://github.com/ymcui/cmrc2018)，英文使用
[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)。二者都有原文、问题和参考答案，
HotpotQA 还提供支持答案的句子标注。数据集卡标注许可证为 CC-BY-SA-4.0；分享数据衍生物时保留来源及许可证。

这里先做文档级检索评测：CMRC 的相关项是问题所属原文，HotpotQA 的相关项是支持句所属文档。
问题及答案单独保存，不写入检索索引。文章被切成 Chunk，但评测时合并同源 Chunk，
取最多 5 份不同来源的文档。找到文档不等于已找到支持句，更不等于最终回答正确。

## 复现

在项目根目录运行 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r benchmarks/retrieval/requirements.txt
.\.venv\Scripts\python.exe benchmarks/retrieval/prepare.py
.\.venv\Scripts\python.exe benchmarks/retrieval/evaluate_bm25.py
```

评测复用项目 `chunk_document()`、`Bm25SparseEncoder`、`Bm25Retriever`、
`QdrantStore.sparse_search()` 和 `project_candidates()`，使用真实 Qdrant 本地内存模式与 IDF。
仅建立 sparse 索引，不生成模拟 dense 向量，不需要 Docker、模型权重或 API Key。
指标由开源库 ranx 计算，并独立核对宏平均 Recall@5。

准备好两个本地模型后，可以用 `run.ps1` 比较四种检索配置。在项目根目录运行：

```powershell
.\benchmarks\retrieval\run.ps1
```

这会依次运行中文和英文数据，每个数据集运行 BM25、Dense、Hybrid 和 Hybrid + Cross-Encoder。
BM25 不需要模型；Dense/Hybrid 需要 BGE-M3；最后一种还需要 `BAAI/bge-reranker-base`。
脚本从 `models/bge-m3` 和 `models/bge-reranker-base` 加载模型，采用 BGE-M3 GPU、重排 CPU、
两者 Batch=1，适用于本机 4GB 显存。使用独立临时 Qdrant，无需启动 Docker 或 HTTP 服务。
脚本设置离线模式，不会重新下载模型。

先检查每种语言的前 3 题（保留原文语料规模），或者分别运行某个数据集：

```powershell
.\benchmarks\retrieval\run.ps1 -MaxQuestions 3
.\benchmarks\retrieval\run.ps1 -Dataset cmrc2018
.\benchmarks\retrieval\run.ps1 -Dataset hotpotqa
```

当前数据仍是每种语言 30 题。`MaxQuestions` 只缩小问题数，不增加抽样规模。
如果只想运行 BM25：

```powershell
.\.venv\Scripts\python.exe benchmarks/retrieval\evaluate_all.py --methods bm25
```

简历用的 100+100 固定子集可以放到独立目录，保留当前 30 题结果：

```powershell
.\.venv\Scripts\python.exe benchmarks/retrieval\prepare.py `
  --root .test-tmp/retrieval-eval-100 --sample-size 100 --seed 42
.\benchmarks\retrieval\run.ps1 `
  -Root .test-tmp/retrieval-eval-100 -MaxQuestions 3
.\benchmarks\retrieval\run.ps1 `
  -Root .test-tmp/retrieval-eval-100
```

这里的 HotpotQA 数据范围是 distractor validation 的前 100 条样本；CMRC2018 使用完整开发集原文，
从中固定抽取 100 道问题。正式结果位于 `retrieval-eval-100/results/`。

已验证环境：Python 3.12.10、torch 2.14.1+cu126、FlagEmbedding 1.4.2、transformers 4.57.6、
ranx 0.3.21、qdrant-client 1.19.1。CUDA 实际可用；两个模型的本地推理已验证。
完整安装版本记录在 `.test-tmp/retrieval-eval/environment-lock.txt`。
执行时直接使用脚本或 `.venv/Scripts/python.exe`；`uv sync` 可能根据项目锁重新安装 CPU torch
并移除仅在评测 requirements 中声明的包。重新安装评测环境可使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[dense]' -r benchmarks/retrieval/requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --timeout 180 torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

## 默认抽样范围

- 每种语言固定随机种子 42，抽取 30 个开发/验证集问题。
- CMRC2018 保留完整开发集原文语料，而非每题只索引它的答案段。
- HotpotQA 保留前 100 个 distractor validation 样本的全部候选原文，合并成共享语料库，
  包括原始干扰文档；从这 100 题中抽取 30 题。不是 fullwiki 评测。
- 原始文件缓存和 SHA-256、种子、范围写入 metadata.json。
- CMRC 原始格式的少量参考答案包含数字/日期单元格异常。保留原始下载，
  转换时只保留文本答案并记录跳过数量，不把数字猜测成日期或答案。
- 未标注的等价证据可能存在，因此结果是基于参考标签的指标。
- 本实验用于学习。调参后需要另留测试集；不能把抽样结果称为官方基准成绩。
- 两个数据集的语言、语料规模和难度不同，不能仅凭分数判断中文与英文谁更好。

## 输出

数据保存在被 git 忽略的 `.test-tmp/retrieval-eval/`：

```text
raw/                          原始下载数据
cmrc2018/ 和 hotpotqa/
  corpus.jsonl                索引原文
  questions.json              问题、参考答案、证据
  questions.md                可直接阅读的 30 道问题与参考答案
  qrels.json                  相关文档标签
  metadata.json               来源、范围、校验值
  documents/                  原文 TXT（便于接入后续入库流程）
  bm25-run.json               文档排名
  bm25-details.json           每题返回片段及命中情况
  bm25-summary.json           BM25 的 Recall/Precision/MRR/NDCG@5 和耗时
  retrieval-summary.json      四种方法的统一对照结果
```

`evaluate_all.py` 会把每种方法的 `*-run.json`、`*-details.json` 和统一的
`retrieval-summary.json` 写入数据集目录。`run.ps1` 把报告写到 `results/cmrc2018/` 和
`results/hotpotqa/`，限制题数的检查写到 `check-N/`，不会覆盖正式报告。这里评测的是检索证据是否找对，尚未评测
LLM 生成答案是否正确；答案生成需要在检索结果之上增加独立的 EM/F1 或答案评测流程。
