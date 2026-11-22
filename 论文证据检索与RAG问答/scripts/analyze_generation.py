"""Score paired generation runs and compare them with paper-cluster bootstrap CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

from scientific_rag.generation_evaluation import evaluate_generation
from scientific_rag.statistics import paired_cluster_bootstrap_ci

METRICS = (
    "answer_exact_match",
    "token_f1",
    "answerability_correct",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "groundedness_per_cited_question",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name.strip(), Path(path.strip())


def unique_prediction_ids(rows: list[dict], method: str) -> set[str]:
    ids = [row.get("question_id") for row in rows]
    if any(not isinstance(question_id, str) for question_id in ids):
        raise ValueError(f"Method {method!r} has a prediction with no valid question_id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Method {method!r} has duplicate question IDs")
    return set(ids)


def select_paired_retrieval_rows(
    rows: list[dict],
    method: str,
    known_question_ids: set[str],
    paired_question_ids: set[str],
) -> list[dict]:
    """Validate the full context file before selecting the generated question subset."""
    question_ids = unique_prediction_ids(rows, f"{method} retrieval")
    unknown = question_ids - known_question_ids
    if unknown:
        raise ValueError(
            f"Retrieval file for method {method!r} contains unknown question IDs: "
            f"{sorted(unknown)[:5]}"
        )
    missing = paired_question_ids - question_ids
    if missing:
        raise ValueError(
            f"Retrieval file for method {method!r} lacks paired questions: "
            f"{sorted(missing)[:5]}"
        )
    return [row for row in rows if row["question_id"] in paired_question_ids]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score saved, question-paired generation runs and compare per-question metrics "
            "with paper-cluster bootstrap intervals. This script makes no model calls."
        )
    )
    parser.add_argument("--question-file", type=Path, default=Path("data/processed/full_validation/evaluation_questions.jsonl"))
    parser.add_argument(
        "--method",
        action="append",
        type=parse_named_path,
        required=True,
        metavar="NAME=JSONL",
        help="saved prediction file; pass at least two methods",
    )
    parser.add_argument(
        "--retrieval",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=JSONL",
        help="matching retrieval context file for citation groundedness",
    )
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("results/experiments/generation_comparison.json")
    )
    args = parser.parse_args()
    if len(args.method) < 2:
        parser.error("at least two --method NAME=JSONL arguments are required")
    if args.context_k <= 0 or args.bootstrap_replicates < 1:
        parser.error("--context-k and --bootstrap-replicates must be positive")

    method_paths = dict(args.method)
    if len(method_paths) != len(args.method):
        parser.error("method names must be unique")
    retrieval_paths = dict(args.retrieval)
    if len(retrieval_paths) != len(args.retrieval):
        parser.error("retrieval method names must be unique")
    unknown_retrievals = set(retrieval_paths) - set(method_paths)
    if unknown_retrievals:
        parser.error(f"retrieval files supplied for unknown methods: {sorted(unknown_retrievals)}")

    question_rows = read_jsonl(args.question_file)
    question_by_id = {row["question_id"]: row for row in question_rows}
    if len(question_by_id) != len(question_rows):
        raise ValueError("Question file contains duplicate question IDs")

    prediction_rows = {name: read_jsonl(path) for name, path in method_paths.items()}
    id_sets = {
        name: unique_prediction_ids(rows, name) for name, rows in prediction_rows.items()
    }
    first_name = next(iter(id_sets))
    expected_ids = id_sets[first_name]
    for name, question_ids in id_sets.items():
        if question_ids != expected_ids:
            raise ValueError(
                f"Methods must contain the same question IDs for paired comparisons; "
                f"{name!r} differs from {first_name!r} by "
                f"{len(question_ids - expected_ids)} extra and "
                f"{len(expected_ids - question_ids)} missing questions"
            )
    missing_questions = expected_ids - set(question_by_id)
    if missing_questions:
        raise ValueError(f"Predictions include unknown question IDs: {sorted(missing_questions)[:5]}")
    paired_questions = [question_by_id[key] for key in sorted(expected_ids)]
    retrieval_rows = {
        name: select_paired_retrieval_rows(
            read_jsonl(path), name, set(question_by_id), expected_ids
        )
        for name, path in retrieval_paths.items()
    }

    method_results = {}
    method_per_question = {}
    for name, rows in prediction_rows.items():
        scored = evaluate_generation(
            paired_questions,
            rows,
            retrieval_predictions=retrieval_rows.get(name),
            context_k=args.context_k,
        )
        method_per_question[name] = scored.pop("per_question")
        scored["models"] = sorted(
            {row.get("model") for row in rows if row.get("model") is not None}
        )
        method_results[name] = scored

    comparisons = {}
    for left_name, right_name in combinations(sorted(method_paths), 2):
        left_rows = method_per_question[left_name]
        right_rows = method_per_question[right_name]
        metric_results = {}
        for metric in METRICS:
            metric_results[metric] = paired_cluster_bootstrap_ci(
                left_rows,
                right_rows,
                metric,
                n_bootstrap=args.bootstrap_replicates,
                seed=args.seed,
            )
        comparisons[f"{left_name}_vs_{right_name}"] = {
            "direction": f"{right_name} minus {left_name}",
            "metrics": metric_results,
        }

    result = {
        "protocol": {
            "python_version": sys.version,
            "question_count": len(paired_questions),
            "paper_cluster_key": "paper_id",
            "bootstrap": {
                "replicates": args.bootstrap_replicates,
                "seed": args.seed,
                "interval": "95% percentile",
            },
            "paired_ids_required": True,
            "difference_direction": "right-hand method minus left-hand method",
            "interpretation_note": (
                "Intervals describe paper-cluster resampling within the supplied question set; "
                "they do not correct deliberate stratified sampling or establish benchmark-wide results."
            ),
            "groundedness_comparison_metric": (
                "Macro mean of each citing question's grounded-citation fraction; the scorer's "
                "aggregate groundedness_micro remains reported separately."
            ),
        },
        "inputs": {
            "question_file": str(args.question_file),
            "question_file_sha256": sha256_file(args.question_file),
            "methods": {
                name: {
                    "prediction_file": str(method_paths[name]),
                    "prediction_file_sha256": sha256_file(method_paths[name]),
                    "retrieval_file": str(retrieval_paths[name]) if name in retrieval_paths else None,
                    "retrieval_file_sha256": (
                        sha256_file(retrieval_paths[name]) if name in retrieval_paths else None
                    ),
                }
                for name in method_paths
            },
        },
        "method_metrics": method_results,
        "per_question_metrics": method_per_question,
        "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "question_count": len(paired_questions),
                "methods": list(method_paths),
                "comparison_count": len(comparisons),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
