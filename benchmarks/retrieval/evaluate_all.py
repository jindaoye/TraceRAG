"""Compare BM25, Dense, Hybrid, and Hybrid + Cross-Encoder retrieval."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import uuid
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient, models
from ranx import Qrels, Run, evaluate

from rag_mcp.chunk_strategies import ChunkingConfig
from rag_mcp.chunking import chunk_document
from rag_mcp.embeddings import BgeM3Embedder
from rag_mcp.rerankers.cross_encoder import CrossEncoderReranker
from rag_mcp.retrievers.dense import DenseRetriever
from rag_mcp.retrievers.hybrid import Bm25Retriever, HybridRetriever
from rag_mcp.sparse import Bm25SparseEncoder
from rag_mcp.storage.qdrant_store import QdrantStore, SPARSE_VECTOR_NAME, DENSE_VECTOR_NAME

DEFAULT_ROOT = ROOT / ".test-tmp" / "retrieval-eval"
METHODS = ("bm25", "dense", "hybrid", "hybrid-rerank")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def unique_documents(candidates, source_ids: dict[str, str], limit: int):
    result, seen = [], set()
    for candidate in candidates:
        document_id = source_ids[candidate.source]
        if document_id in seen:
            continue
        seen.add(document_id)
        result.append((document_id, candidate))
        if len(result) == limit:
            break
    return result


def _load_dataset(folder: Path, size: int, overlap: int):
    corpus = [json.loads(line) for line in (folder / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
    qrels = json.loads((folder / "qrels.json").read_text(encoding="utf-8"))
    sources = folder / "documents"
    sources.mkdir(exist_ok=True)
    config = ChunkingConfig("character", size, overlap)
    chunks, source_ids = [], {}
    for document in corpus:
        filename = document["id"] + ".txt"
        source_ids[filename] = document["id"]
        path = sources / filename
        path.write_text(document["title"] + "\n\n" + document["text"], encoding="utf-8")
        chunks.extend(chunk_document(path, source_root=sources, config=config))
    return corpus, questions, qrels, chunks, source_ids


def _build_store(chunks, *, embedder, encoder):
    dense_vectors = embedder.embed_documents([chunk.text for chunk in chunks]) if embedder else None
    sparse_vectors = encoder.encode_documents([chunk.text for chunk in chunks])
    client = QdrantClient(":memory:")
    store = QdrantStore(collection="learning_eval", client=client)
    store.ensure_collection()
    points = []
    for index, (chunk, sparse) in enumerate(zip(chunks, sparse_vectors, strict=True)):
        vectors = {SPARSE_VECTOR_NAME: models.SparseVector(
            indices=list(sparse.indices), values=list(sparse.values)
        )}
        if dense_vectors is not None:
            vectors[DENSE_VECTOR_NAME] = list(dense_vectors[index])
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
            vector=vectors,
            payload={
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "title": chunk.title,
                "metadata": chunk.metadata,
            },
        ))
    for start in range(0, len(points), 256):
        client.upsert("learning_eval", points[start:start + 256], wait=True)
    return client, store


def _run_method(method, questions, qrels, retriever, reranker, source_ids, candidate_k, top_k):
    run, details = {}, []
    # Exclude lazy model loading and first inference from per-query latency.
    warm_candidates = retriever.search(questions[0]["question"], limit=candidate_k, filters={})
    if reranker is not None:
        reranker.rerank(questions[0]["question"], warm_candidates[:1])
    for number, question in enumerate(questions, 1):
        retrieval_started = perf_counter()
        candidates = retriever.search(question["question"], limit=candidate_k, filters={})
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        rerank_ms = 0.0
        if reranker is not None:
            rerank_started = perf_counter()
            candidates = reranker.rerank(question["question"], candidates)
            rerank_ms = (perf_counter() - rerank_started) * 1000
        ranked = unique_documents(candidates, source_ids, top_k)
        run[question["id"]] = {
            document_id: float(len(ranked) - rank)
            for rank, (document_id, _candidate) in enumerate(ranked)
        }
        expected = set(qrels[question["id"]])
        retrieved = {document_id for document_id, _candidate in ranked}
        details.append({
            "id": question["id"],
            "question": question["question"],
            "reference_answers": question["answers"],
            "relevant_documents": sorted(expected),
            "evidence": question["evidence"],
            f"reference_recall_at_{top_k}": len(expected & retrieved) / len(expected),
            "all_reference_documents_found": expected <= retrieved,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "results": [
                {
                    "document_id": document_id,
                    "chunk_id": candidate.chunk_id,
                    "score": candidate.score,
                    "retrieval_score": candidate.metadata.get("retrieval_score"),
                    "text": candidate.text,
                }
                for document_id, candidate in ranked
            ],
        })
        if number == 1 or number % 10 == 0 or number == len(questions):
            print(f"  {method}: {number}/{len(questions)} questions", flush=True)
    measures = [f"{metric}@{top_k}" for metric in ("recall", "precision", "mrr", "ndcg")]
    scores = evaluate(Qrels(qrels), Run(run), measures, threads=1)
    manual_recall = mean(row[f"reference_recall_at_{top_k}"] for row in details)
    if abs(float(scores[f"recall@{top_k}"]) - manual_recall) > 1e-10:
        raise ValueError("ranx recall differs from independent calculation")
    summary = {
        "method": method,
        "questions": len(questions),
        "metrics": {key: float(value) for key, value in scores.items()},
        "mean_retrieval_ms": mean(row["retrieval_ms"] for row in details),
        "mean_rerank_ms": mean(row["rerank_ms"] for row in details),
        "all_reference_documents_found_rate": mean(row["all_reference_documents_found"] for row in details),
        "limitations": [
            "Document-level retrieval evaluation; it does not score generated answer correctness.",
            "HotpotQA reference documents are a proxy for supporting evidence; sentence coverage is not measured.",
            "Latency uses in-memory local Qdrant and depends on the local model device.",
        ],
    }
    return run, details, summary


def run_dataset(folder: Path, methods: list[str], size: int, overlap: int, candidate_k: int,
                top_k: int, embedding_model: str, embedding_device: str | None,
                reranker_model: str, reranker_device: str | None,
                embedding_batch_size: int = 1, reranker_batch_size: int = 1,
                max_questions: int | None = None, output_root: Path | None = None):
    output_folder = output_root / folder.name if output_root else folder
    output_folder.mkdir(parents=True, exist_ok=True)
    corpus, questions, qrels, chunks, source_ids = _load_dataset(folder, size, overlap)
    if max_questions is not None:
        questions = questions[:max_questions]
        qrels = {question["id"]: qrels[question["id"]] for question in questions}
    if not questions:
        raise ValueError("No evaluation questions")
    need_dense = any(method in {"dense", "hybrid", "hybrid-rerank"} for method in methods)
    embedder = BgeM3Embedder(embedding_model, device=embedding_device,
                           batch_size=embedding_batch_size) if need_dense else None
    encoder = Bm25SparseEncoder()
    print(f"  Building index: {len(corpus)} documents, {len(chunks)} chunks", flush=True)
    index_started = perf_counter()
    client, store = _build_store(chunks, embedder=embedder, encoder=encoder)
    try:
        retrievers = {"bm25": Bm25Retriever(encoder, store)}
        if need_dense:
            retrievers["dense"] = DenseRetriever(embedder, store)
            retrievers["hybrid"] = HybridRetriever(embedder, encoder, store)
        reranker = None
        if "hybrid-rerank" in methods:
            reranker = CrossEncoderReranker(
                reranker_model, device=reranker_device, max_candidates=candidate_k,
                batch_size=reranker_batch_size,
            )
        output = {
            "dataset": folder.name,
            "documents": len(corpus),
            "chunks": len(chunks),
            "chunking": {"strategy": "character", "size": size, "overlap": overlap},
            "candidate_k_chunks": candidate_k,
            "top_k_unique_documents": top_k,
            "embedding_model": embedding_model if need_dense else None,
            "reranker_model": reranker_model if "hybrid-rerank" in methods else None,
            "embedding_device": embedding_device,
            "reranker_device": reranker_device,
            "embedding_batch_size": embedding_batch_size,
            "reranker_batch_size": reranker_batch_size,
            "index_seconds": perf_counter() - index_started,
            "max_questions": max_questions,
            "latency_policy": "Model loading and warmup excluded; query encoding included",
            "methods": {},
            "versions": {
                **{name: importlib.metadata.version(name) for name in ("ranx", "qdrant-client")},
                **({name: importlib.metadata.version(name) for name in
                    ("torch", "FlagEmbedding", "transformers")} if need_dense else {}),
            },
        }
        for method in methods:
            base_method = "hybrid" if method == "hybrid-rerank" else method
            method_reranker = reranker if method == "hybrid-rerank" else None
            run, details, summary = _run_method(
                method, questions, qrels, retrievers[base_method], method_reranker,
                source_ids, candidate_k, top_k,
            )
            output["methods"][method] = summary
            write_json(output_folder / f"{method}-run.json", run)
            write_json(output_folder / f"{method}-details.json", details)
            write_json(output_folder / f"{method}-summary.json", summary)
            write_json(output_folder / "retrieval-summary.json", output)
        write_json(output_folder / "retrieval-summary.json", output)
        print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
        return output
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, help="Save reports separately from input data")
    parser.add_argument("--dataset", choices=["cmrc2018", "hotpotqa", "all"], default="all")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--embedding-device", default=os.getenv("RAG_EMBEDDING_DEVICE"))
    parser.add_argument("--reranker-model", default=os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base"))
    parser.add_argument("--reranker-device", default=os.getenv("RAG_RERANKER_DEVICE"))
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--max-questions", type=int, help="Limit questions; retain the full prepared corpus")
    args = parser.parse_args()
    if args.candidate_k < args.top_k:
        parser.error("candidate-k must be at least top-k")
    if not 1 <= args.top_k <= args.candidate_k <= 100:
        parser.error("require 1 <= top-k <= candidate-k <= 100")
    if args.embedding_batch_size < 1 or not 1 <= args.reranker_batch_size <= 64:
        parser.error("embedding batch must be positive; reranker batch must be 1..64")
    if args.max_questions is not None and args.max_questions < 1:
        parser.error("max-questions must be positive")
    names = ["cmrc2018", "hotpotqa"] if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"Evaluating {name}: {', '.join(args.methods)}", flush=True)
        run_dataset(
            args.root / name, args.methods, args.chunk_size, args.overlap,
            args.candidate_k, args.top_k, args.embedding_model, args.embedding_device,
            args.reranker_model, args.reranker_device,
            args.embedding_batch_size, args.reranker_batch_size, args.max_questions, args.output_root,
        )


if __name__ == "__main__":
    main()
