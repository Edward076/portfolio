"""Evaluate already-saved answer and citation predictions without calling a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scientific_rag.generation_evaluation import evaluate_generation


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score saved Qasper answer and citation predictions; this script makes no API calls."
    )
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument(
        "--question-file",
        type=Path,
        default=Path("data/processed/full_validation/evaluation_questions.jsonl"),
    )
    parser.add_argument("--retrieval-file", type=Path)
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiments/full_validation/generation_metrics.json"),
    )
    args = parser.parse_args()

    retrieval = read_jsonl(args.retrieval_file) if args.retrieval_file else None
    result = evaluate_generation(
        read_jsonl(args.question_file),
        read_jsonl(args.prediction_file),
        retrieval_predictions=retrieval,
        context_k=args.context_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result}, indent=2))


if __name__ == "__main__":
    main()
