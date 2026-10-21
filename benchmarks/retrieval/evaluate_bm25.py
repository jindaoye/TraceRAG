"""Evaluate project BM25 on source-document labels, using isolated in-memory Qdrant."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import uuid
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient, models
from ranx import Qrels, Run, evaluate

from rag_mcp.chunking import chunk_document
from rag_mcp.chunk_strategies import ChunkingConfig
from rag_mcp.retrievers.hybrid import Bm25Retriever
from rag_mcp.sparse import Bm25SparseEncoder
from rag_mcp.storage.qdrant_store import QdrantStore, SPARSE_VECTOR_NAME

DEFAULT_ROOT = ROOT / ".test-tmp" / "retrieval-eval"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def unique_documents(candidates, source_ids, limit):
    result, seen = [], set()
    for candidate in candidates:
        docid = source_ids[candidate.source]
        if docid in seen:
            continue
        seen.add(docid)
        result.append((docid, candidate))
        if len(result) == limit:
            break
    return result


def run_dataset(folder, size, overlap, candidate_k):
    corpus = [json.loads(line) for line in (folder / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
    qrels = json.loads((folder / "qrels.json").read_text(encoding="utf-8"))
    sources = folder / "documents"
    sources.mkdir(exist_ok=True)
    chunks, source_ids = [], {}
    config = ChunkingConfig("character", size, overlap)
    for doc in corpus:
        filename = doc["id"] + ".txt"
        source_ids[filename] = doc["id"]
        path = sources / filename
        path.write_text(doc["title"] + "\n\n" + doc["text"], encoding="utf-8")
        chunks.extend(chunk_document(path, source_root=sources, config=config))
    encoder = Bm25SparseEncoder()
    sparse_vectors = encoder.encode_documents([c.text for c in chunks])
    client = QdrantClient(":memory:")
    try:
        # A real sparse-only index: no dummy dense vectors, no model API calls.
        client.create_collection("learning_eval", vectors_config={}, sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        })
        points = []
        for chunk, vector in zip(chunks, sparse_vectors, strict=True):
            if not vector.indices:
                continue
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector={SPARSE_VECTOR_NAME: models.SparseVector(indices=list(vector.indices), values=list(vector.values))},
                payload={"document_id": chunk.document_id, "chunk_id": chunk.chunk_id,
                         "text": chunk.text, "source": chunk.source, "title": chunk.title,
                         "metadata": chunk.metadata},
            ))
        for start in range(0, len(points), 256):
            client.upsert("learning_eval", points[start:start + 256], wait=True)
        store = QdrantStore(collection="learning_eval", client=client)
        retriever = Bm25Retriever(encoder, store)
        runs, details = {}, []
        for question in questions:
            started = perf_counter()
            candidates = retriever.search(question["question"], limit=candidate_k, filters={})
            latency_ms = (perf_counter() - started) * 1000
            ranked = unique_documents(candidates, source_ids, 5)
            # Preserve actual returned order, including ties, with rank scores.
            runs[question["id"]] = {docid: float(len(ranked) - i) for i, (docid, _) in enumerate(ranked)}
            expected = set(qrels[question["id"]])
            retrieved = {docid for docid, _ in ranked}
            details.append({
                "id": question["id"], "question": question["question"],
                "reference_answers": question["answers"],
                "relevant_documents": sorted(expected), "evidence": question["evidence"],
                "reference_recall_at_5": len(expected & retrieved) / len(expected),
                "all_reference_documents_found": expected <= retrieved,
                "latency_ms": latency_ms,
                "results": [{"document_id": docid, "chunk_id": c.chunk_id,
                             "score": c.score, "text": c.text} for docid, c in ranked],
            })
        measures = ["recall@5", "precision@5", "mrr@5", "ndcg@5"]
        scores = evaluate(Qrels(qrels), Run(runs), measures, threads=1)
        # Independently verify the core metric to catch ID/rank conversion errors.
        manual_recall = mean(row["reference_recall_at_5"] for row in details)
        if abs(float(scores["recall@5"]) - manual_recall) > 1e-10:
            raise ValueError("ranx recall does not match the per-query calculation")
        output = {
            "dataset": folder.name, "method": "project BM25-style sparse + Qdrant IDF",
            "questions": len(questions), "documents": len(corpus), "chunks": len(chunks),
            "chunking": {"strategy": "character", "size": size, "overlap": overlap},
            "candidate_k_chunks": candidate_k, "top_k_unique_documents": 5,
            "metrics": {key: float(value) for key, value in scores.items()},
            "mean_retrieval_ms": mean(row["latency_ms"] for row in details),
            "all_reference_documents_found_rate": mean(row["all_reference_documents_found"] for row in details),
            "versions": {name: importlib.metadata.version(name) for name in ["ranx", "qdrant-client"]},
            "limitations": ["Sampled document-level retrieval evaluation; no generated answers were scored.",
                             "Latency is for in-memory local Qdrant, not production HTTP service.",
                             "Recall measures reference documents, not answer quality or supporting-sentence coverage."],
        }
        write_json(folder / "bm25-run.json", runs)
        write_json(folder / "bm25-details.json", details)
        write_json(folder / "bm25-summary.json", output)
        print(json.dumps(output, ensure_ascii=True, indent=2), flush=True)
        return output
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", choices=["cmrc2018", "hotpotqa", "all"], default="all")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=30)
    args = parser.parse_args()
    if args.candidate_k < 5:
        parser.error("candidate-k must be at least 5")
    names = ["cmrc2018", "hotpotqa"] if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"Evaluating {name}", flush=True)
        run_dataset(args.root / name, args.chunk_size, args.overlap, args.candidate_k)


if __name__ == "__main__":
    main()
