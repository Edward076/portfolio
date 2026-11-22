from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scientific_rag.evaluation import retrieval_metrics


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect development-set top-K and chunking ablations."
    )
    parser.add_argument("--topk-root", type=Path, default=Path("results/ablations/topk"))
    parser.add_argument("--experiment-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--output", type=Path, default=Path("results/ablation_results.csv"))
    args = parser.parse_args()
    rows = []

    for prediction_path in sorted(args.topk_root.rglob("predictions.jsonl")):
        if "dev" not in prediction_path.parts:
            continue
        for prediction in read_jsonl(prediction_path):
            retrieved = prediction.get("retrieved", [])
            retrieved_paragraphs = [
                item.get("paragraph_id", item["chunk_id"]) for item in retrieved
            ]
            evidence_sets = [set(group) for group in prediction.get("gold_evidence_sets", [])]
            method = prediction.get("method", prediction_path.parts[-4])
            for k in (3, 5, 10):
                metrics = retrieval_metrics(retrieved_paragraphs, evidence_sets, k=k)
                rows.append(
                    {
                        "study": "top_k",
                        "method": method,
                        "parameter": k,
                        "parameter_name": "final_k",
                        "chunking": "token-aware",
                        "question_id": prediction["question_id"],
                        "paper_id": prediction["paper_id"],
                        **metrics,
                    }
                )

    for method in ("bm25", "dense"):
        for chunking in ("paragraph", "token-aware"):
            metric_path = args.experiment_root / method / "dev" / "retrieval_metrics.csv"
            if chunking == "token-aware":
                metric_path = (
                    args.experiment_root / method / "dev" / "token-aware" / "retrieval_metrics.csv"
                )
            if not metric_path.exists():
                continue
            with metric_path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    rows.append(
                        {
                            "study": "chunking",
                            "method": method,
                            "parameter": chunking,
                            "parameter_name": "chunking",
                            "chunking": chunking,
                            "question_id": row["question_id"],
                            "paper_id": row["paper_id"],
                            **{
                                key: row[key]
                                for key in (
                                    "evidence_recall",
                                    "mrr",
                                    "hit_rate",
                                    "complete_evidence_coverage",
                                )
                            },
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "study",
        "method",
        "parameter_name",
        "parameter",
        "chunking",
        "question_id",
        "paper_id",
        "evidence_recall",
        "mrr",
        "hit_rate",
        "complete_evidence_coverage",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} per-question ablation rows to {args.output}")


if __name__ == "__main__":
    main()
