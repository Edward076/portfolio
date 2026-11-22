from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze(prediction_path: Path, question_path: Path, context_k: int = 5) -> list[dict]:
    questions = {row["question_id"]: row for row in read_jsonl(question_path)}
    errors = []
    predictions = read_jsonl(prediction_path)
    for prediction in predictions:
        question = questions.get(prediction["question_id"])
        if question is None:
            continue
        answerable = [item for item in question["annotations"] if not item["unanswerable"]]
        evidence_sets = [set(item["mapped_evidence_ids"]) for item in answerable]
        mapped_sets = [evidence for evidence in evidence_sets if evidence]
        retrieved = prediction.get("retrieved", [])
        final_ids = [item.get("paragraph_id", item["chunk_id"]) for item in retrieved[:context_k]]
        candidate_items = retrieved
        if retrieved and all("pre_rerank_rank" in item for item in retrieved):
            candidate_items = sorted(retrieved, key=lambda item: item["pre_rerank_rank"])
        candidate_ids = [item.get("paragraph_id", item["chunk_id"]) for item in candidate_items]
        if not answerable:
            category = "data_annotation_limitation: fully unanswerable question"
        elif not mapped_sets:
            category = "data_annotation_limitation: no mapped evidence"
        elif set(final_ids) & set().union(*mapped_sets):
            continue
        elif set(candidate_ids) & set().union(*mapped_sets):
            category = "ranking_failure"
        else:
            category = "retrieval_failure"
        candidate_set = set(candidate_ids)
        best_candidate_recall = (
            max(len(candidate_set & gold) / len(gold) for gold in mapped_sets)
            if mapped_sets
            else None
        )
        errors.append(
            {
                "question_id": question["question_id"],
                "paper_id": question["paper_id"],
                "question": question["question"],
                "method": prediction.get("method", "unknown"),
                "error_category": category,
                "candidate_depth": len(candidate_ids),
                "candidate_evidence_recall": best_candidate_recall,
                "context_k": context_k,
                "context_paragraph_ids": json.dumps(final_ids),
                "candidate_paragraph_ids": json.dumps(candidate_ids),
                "gold_evidence_sets": json.dumps([sorted(gold) for gold in mapped_sets]),
                "annotator_count": len(question["annotations"]),
            }
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify genuine retrieval and ranking errors.")
    parser.add_argument(
        "--prediction-root", type=Path, default=Path("results/experiments/candidates20")
    )
    parser.add_argument(
        "--question-file", type=Path, default=Path("data/processed/evaluation_questions.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("results/error_analysis.csv"))
    parser.add_argument("--context-k", type=int, default=5)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.prediction_root.rglob("predictions.jsonl")):
        if "evaluation" not in path.parts:
            continue
        rows.extend(analyze(path, args.question_file, args.context_k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["question_id", "paper_id", "method", "error_category"]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} observed failures to {args.output}")


if __name__ == "__main__":
    main()
