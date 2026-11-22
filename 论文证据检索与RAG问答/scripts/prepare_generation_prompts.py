from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scientific_rag.generation_prompt import (
    build_closed_book_messages,
    build_generation_messages,
)

RAG_METHODS = ("dense", "hybrid", "hybrid_reranked")
METHODS = ("closed_book", *RAG_METHODS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def answerability_stratum(question: dict[str, Any]) -> str:
    annotations = question.get("annotations", [])
    answerable = [item for item in annotations if not item.get("unanswerable", False)]
    if not annotations:
        return "no_annotations"
    if not answerable:
        return "fully_unanswerable"
    if len(answerable) != len(annotations):
        return "mixed_answerability"
    if any(item.get("mapped_evidence_ids") for item in answerable):
        return "answerable_mapped_evidence"
    return "answerable_unmapped_evidence"


def select_balanced_questions(
    questions: list[dict[str, Any]], per_stratum: int, seed: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        groups.setdefault(answerability_stratum(question), []).append(question)

    selected = []
    counts = {}
    for stratum, group in sorted(groups.items()):
        group.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{stratum}:{row['question_id']}".encode()
            ).hexdigest()
        )
        chosen = group[:per_stratum]
        counts[stratum] = len(chosen)
        selected.extend({**row, "_sampling_stratum": stratum} for row in chosen)
    selected.sort(key=lambda row: (row["_sampling_stratum"], row["question_id"]))
    return selected, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced, offline prompt batch; this script makes no API calls."
    )
    parser.add_argument(
        "--question-file",
        type=Path,
        default=Path("data/processed/full_validation/evaluation_questions.jsonl"),
    )
    parser.add_argument(
        "--context-root", type=Path, default=Path("results/experiments/generation_contexts")
    )
    parser.add_argument("--per-stratum", type=int, default=5)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/experiments/generation_pilot_preflight")
    )
    args = parser.parse_args()
    if args.per_stratum < 1 or args.context_k < 1:
        parser.error("--per-stratum and --context-k must be positive")

    questions = read_jsonl(args.question_file)
    if len({row["question_id"] for row in questions}) != len(questions):
        raise ValueError("Question file contains duplicate question IDs")
    selected, selected_counts = select_balanced_questions(
        questions, per_stratum=args.per_stratum, seed=args.seed
    )
    selected_ids = {row["question_id"] for row in selected}
    retrieval_by_method = {}
    for method in RAG_METHODS:
        path = args.context_root / method / "evaluation/token-aware/predictions.jsonl"
        rows = read_jsonl(path)
        by_id = {row["question_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Duplicate question IDs in retrieval file: {path}")
        missing = selected_ids - by_id.keys()
        if missing:
            raise ValueError(f"Retrieval file {path} lacks selected questions: {sorted(missing)}")
        retrieval_by_method[method] = by_id

    jobs = []
    for question in selected:
        question_id = question["question_id"]
        for method in METHODS:
            if method == "closed_book":
                prompt = build_closed_book_messages(
                    question["question"], question.get("paper_title", "")
                )
                context_passages = []
            else:
                retrieval = retrieval_by_method[method][question_id]
                context_passages = retrieval.get("retrieved", [])[: args.context_k]
                prompt = build_generation_messages(
                    question["question"], context_passages, context_k=args.context_k
                )
            jobs.append(
                {
                    "question_id": question_id,
                    "paper_id": question["paper_id"],
                    "method": method,
                    "sampling_stratum": question["_sampling_stratum"],
                    "prompt_version": prompt["prompt_version"],
                    "allowed_citations": prompt["allowed_citations"],
                    "messages": prompt["messages"],
                    "retrieved": context_passages,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = args.output_dir / "prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8", newline="\n") as stream:
        for job in jobs:
            stream.write(json.dumps(job, ensure_ascii=False) + "\n")

    manifest = {
        "status": "prepared_only_no_api_calls",
        "api_calls_made": 0,
        "question_file": str(args.question_file),
        "question_count": len(questions),
        "seed": args.seed,
        "questions_per_stratum": args.per_stratum,
        "selected_question_count": len(selected),
        "selected_counts_by_stratum": selected_counts,
        "selected_question_ids_by_stratum": {
            stratum: [
                row["question_id"]
                for row in selected
                if row["_sampling_stratum"] == stratum
            ]
            for stratum in sorted(selected_counts)
        },
        "methods": list(METHODS),
        "job_count": len(jobs),
        "context_k": args.context_k,
        "prompts_file": prompts_path.name,
    }
    manifest_path = args.output_dir / "preflight_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
