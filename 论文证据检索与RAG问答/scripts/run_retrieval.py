from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

from scientific_rag.bm25_retriever import BM25Retriever
from scientific_rag.chunking import split_token_aware
from scientific_rag.evaluation import aggregate_metrics, evaluate_retrieval

RUNTIME_DISTRIBUTIONS = (
    "datasets",
    "numpy",
    "pandas",
    "rank-bm25",
    "sentence-transformers",
    "chromadb",
    "torch",
    "transformers",
    "scikit-learn",
)


def installed_package_versions() -> dict[str, str | None]:
    """Return versions for core and optional retrieval dependencies."""
    versions = {}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qasper paragraph retrieval baselines.")
    parser.add_argument("--split", choices=("dev", "evaluation"), default="evaluation")
    parser.add_argument(
        "--method", choices=("bm25", "dense", "hybrid", "hybrid_reranked"), default="bm25"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-candidate-k", type=int, default=20)
    parser.add_argument("--bm25-candidate-k", type=int, default=20)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )
    parser.add_argument("--embedding-revision", default="b207367332321f8e44f96e224ef15bc607f4dbf0")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--reranker-revision", default="233902d25c440f23af6f7d6e94d2946bac0bee0a")
    parser.add_argument("--chunking", choices=("paragraph", "token-aware"), default="paragraph")
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-unscored-predictions",
        action="store_true",
        help="retrieve every question for downstream generation while scoring mapped-evidence cases only",
    )
    args = parser.parse_args()

    chunks = read_jsonl(args.data_dir / f"{args.split}_chunks.jsonl")
    questions = read_jsonl(args.data_dir / f"{args.split}_questions.jsonl")
    if args.chunking == "token-aware":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.embedding_model, revision=args.embedding_revision
        )
        tokenizer.model_max_length = max(args.max_tokens + 1, 1_000_000)
        chunks = split_token_aware(
            chunks, tokenizer, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens
        )
    bm25 = BM25Retriever(chunks)
    retriever = bm25
    if args.method in {"dense", "hybrid", "hybrid_reranked"}:
        from scientific_rag.dense_retriever import DenseRetriever

        dense = DenseRetriever(
            chunks, model_name=args.embedding_model, model_revision=args.embedding_revision
        )
        retriever = dense
        if args.method in {"hybrid", "hybrid_reranked"}:
            from scientific_rag.hybrid_retriever import HybridRetriever

            hybrid = HybridRetriever(
                dense,
                bm25,
                dense_top_k=args.dense_candidate_k,
                bm25_top_k=args.bm25_candidate_k,
                rrf_constant=args.rrf_constant,
            )
            retriever = hybrid
            if args.method == "hybrid_reranked":
                from scientific_rag.reranker import CrossEncoderReranker, RerankedHybridRetriever

                retriever = RerankedHybridRetriever(
                    hybrid,
                    CrossEncoderReranker(args.reranker_model, args.reranker_revision),
                    candidate_top_k=20,
                )

    rows, predictions = evaluate_retrieval(
        questions,
        retriever,
        top_k=args.top_k,
        include_unscored_predictions=args.include_unscored_predictions,
    )
    for row in rows:
        row["chunking"] = args.chunking
    for prediction in predictions:
        prediction["chunking"] = args.chunking
    output_dir = args.output_dir or Path("results/experiments") / args.method
    output_dir = output_dir / args.split
    if args.chunking != "paragraph":
        output_dir = output_dir / args.chunking
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "retrieval_metrics.csv", rows)
    latency_rows = [
        {key: row[key] for key in row if key.endswith("latency_ms")}
        | {"question_id": row["question_id"], "paper_id": row["paper_id"], "method": args.method}
        for row in rows
    ]
    write_csv(output_dir / "latency_metrics.csv", latency_rows)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            prediction["method"] = args.method
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    summary = aggregate_metrics(rows)
    latency_values = [row["retrieval_latency_ms"] for row in rows]
    latency_sorted = sorted(latency_values)
    summary.update(
        {
            "method": args.method,
            "paper_count": len({question["paper_id"] for question in questions}),
            "all_question_count": len(questions),
            "source_chunks": len(chunks),
            "latency_ms_mean": sum(latency_values) / len(latency_values)
            if latency_values
            else None,
            "latency_ms_median": latency_sorted[len(latency_sorted) // 2]
            if latency_sorted
            else None,
            "latency_ms_p95": (
                latency_sorted[min(int(len(latency_sorted) * 0.95), len(latency_sorted) - 1)]
                if latency_sorted
                else None
            ),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = args.data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_key = "development" if args.split == "dev" else "evaluation"
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "split": args.split,
                "top_k": args.top_k,
                "dense_candidate_k": args.dense_candidate_k,
                "bm25_candidate_k": args.bm25_candidate_k,
                "rrf_constant": args.rrf_constant,
                "embedding_model": args.embedding_model,
                "embedding_revision": args.embedding_revision,
                "reranker_model": args.reranker_model,
                "reranker_revision": args.reranker_revision,
                "chunking": args.chunking,
                "max_tokens": args.max_tokens if args.chunking == "token-aware" else None,
                "overlap_tokens": args.overlap_tokens if args.chunking == "token-aware" else None,
                "device": "cpu",
                "dataset_revision": manifest["source_revision"],
                "parquet_conversion_revision": manifest.get("parquet_conversion_revision"),
                "sample_fingerprint": manifest[sample_key]["sample_fingerprint_sha256"],
                "include_unscored_predictions": args.include_unscored_predictions,
                "python_version": __import__("sys").version,
                "package_versions": installed_package_versions(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"split": args.split, **summary, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
