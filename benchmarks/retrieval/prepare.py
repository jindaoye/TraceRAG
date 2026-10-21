"""Download labelled QA data and create reproducible bilingual learning subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import requests

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / ".test-tmp" / "retrieval-eval"
CMRC_URL = "https://raw.githubusercontent.com/ymcui/cmrc2018/master/data/cmrc2018_dev.json"
HOTPOT_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa"
    "&config=distractor&split=validation&offset=0&length=100"
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {path.name}", flush=True)
        response = requests.get(url, timeout=(15, 120))
        response.raise_for_status()
        response.json()  # Reject a non-JSON error page before caching it.
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return json.loads(path.read_text(encoding="utf-8"))


def cmrc_records(raw):
    corpus, questions = [], []
    for item in raw:
        docid = item["context_id"]
        context = item["context_text"]
        corpus.append({"id": docid, "title": item["title"], "text": context})
        for qa in item["qas"]:
            # The original-format dev file contains numeric/date-cell artifacts
            # among alternatives. Keep only actual text references, not guessed strings.
            answers = list(dict.fromkeys(a for a in qa["answers"] if isinstance(a, str)))
            if not answers or not any(answer and answer in context for answer in answers):
                continue
            questions.append({
                "id": qa["query_id"], "question": qa["query_text"],
                "answers": answers, "relevant_documents": [docid],
                "evidence": [{"document_id": docid, "answer_spans": answers}],
                "non_text_answers_skipped": sum(not isinstance(a, str) for a in qa["answers"]),
            })
    return corpus, questions


def hotpot_records(raw):
    corpus, questions = {}, []
    for entry in raw["rows"]:
        item = entry["row"]
        context = item["context"]
        local = {}
        for title, sentences in zip(context["title"], context["sentences"], strict=True):
            text = "".join(sentences)
            # Titles may have different passage versions; never overwrite them.
            docid = "hp-" + hashlib.sha256((title + "\0" + text).encode()).hexdigest()[:24]
            corpus[docid] = {"id": docid, "title": title, "text": text}
            local[title] = (docid, sentences)
        facts = item["supporting_facts"]
        evidence = []
        for title, sent_id in zip(facts["title"], facts["sent_id"], strict=True):
            docid, sentences = local[title]
            if not 0 <= sent_id < len(sentences):
                raise ValueError(f"Invalid supporting sentence in {item['id']}")
            evidence.append({"document_id": docid, "title": title,
                             "sentence_id": sent_id, "sentence": sentences[sent_id]})
        if not evidence or not item["answer"]:
            raise ValueError(f"Unlabelled question: {item['id']}")
        questions.append({
            "id": item["id"], "question": item["question"],
            "answers": [item["answer"]],
            "relevant_documents": list(dict.fromkeys(e["document_id"] for e in evidence)),
            "evidence": evidence,
        })
    return list(corpus.values()), questions


def export(root, name, corpus, questions, *, sample_size, seed, url, raw_path, license_name):
    if not 1 <= sample_size <= len(questions):
        raise ValueError(f"sample-size must be between 1 and {len(questions)} for {name}")
    selected = random.Random(seed).sample(questions, sample_size)
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    documents = {doc["id"]: doc for doc in corpus}
    if len(documents) != len(corpus):
        raise ValueError("Duplicate corpus IDs")
    for question in selected:
        if not set(question["relevant_documents"]) <= documents.keys():
            raise ValueError("A relevant document is missing from the corpus")
    with (folder / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for doc in corpus:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    write_json(folder / "questions.json", selected)
    preview = [f"# {name}：问题与参考答案", "",
               "下面是公开数据集的参考答案；不是模型生成结果。相关文档标签用于检索评测。", ""]
    for index, question in enumerate(selected, start=1):
        preview.extend([f"## {index}. {question['question']}", "",
                        "参考答案：" + " / ".join(question["answers"]), "",
                        f"问题 ID：`{question['id']}`", "",
                        "相关原文：" + ", ".join(f"`{docid}`" for docid in question["relevant_documents"]), ""])
        for evidence in question["evidence"]:
            if "sentence" in evidence:
                preview.extend([f"支持句（{evidence['title']}，句子 {evidence['sentence_id']}）：", "",
                                evidence["sentence"], ""])
    (folder / "questions.md").write_text("\n".join(preview), encoding="utf-8")
    write_json(folder / "qrels.json", {
        q["id"]: {docid: 1 for docid in q["relevant_documents"]} for q in selected
    })
    write_json(folder / "metadata.json", {
        "dataset": name, "source_url": url, "license": license_name,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "seed": seed, "sample_size": sample_size, "available_questions": len(questions),
        "corpus_documents": len(corpus), "evaluation_unit": "source passage/document",
        "non_text_answers_skipped_in_sample": sum(q.get("non_text_answers_skipped", 0) for q in selected),
        "non_text_answers_skipped_in_available_questions": sum(q.get("non_text_answers_skipped", 0) for q in questions),
        "scope": ("Complete CMRC2018 development passage corpus" if name == "cmrc2018"
                  else "Union of contexts from first 100 HotpotQA distractor validation rows"),
        "limitations": [
            "Learning subset; not an official benchmark score.",
            "Reference labels may omit equivalent evidence in other documents.",
            "Document recall does not measure answer correctness or supporting-sentence coverage.",
            "Keep this subset fixed; use separate held-out data for final claims after tuning.",
        ],
    })
    print(f"{name}: {len(selected)} questions, {len(corpus)} corpus documents -> {folder}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sources = [
        ("cmrc2018", CMRC_URL, cmrc_records, "CC-BY-SA-4.0"),
        ("hotpotqa", HOTPOT_URL, hotpot_records, "CC-BY-SA-4.0"),
    ]
    for name, url, convert, license_name in sources:
        path = args.root / "raw" / f"{name}.json"
        raw = download(url, path)
        corpus, questions = convert(raw)
        export(args.root, name, corpus, questions, sample_size=args.sample_size,
               seed=args.seed, url=url, raw_path=path, license_name=license_name)


if __name__ == "__main__":
    main()
