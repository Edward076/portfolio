from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METHODS = ("bm25", "dense", "hybrid", "hybrid_reranked")
METRICS = ("evidence_recall", "mrr", "hit_rate", "complete_evidence_coverage")
PROVENANCE_FIELDS = (
    "dataset_revision",
    "parquet_conversion_revision",
    "sample_fingerprint",
    "embedding_revision",
    "reranker_revision",
    "method",
    "split",
    "top_k",
    "chunking",
    "max_tokens",
    "overlap_tokens",
)
PREDICTION_FIELDS = (
    "ranked_parent_paragraph_ids_match",
    "ranked_chunk_ids_match",
    "ranked_text_match",
    "retrieval_scores_match",
    "gold_evidence_sets_match",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _unique_by_question(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in records:
            raise ValueError(f"Duplicate question_id {question_id!r} in {path}")
        records[question_id] = row
    return records


def _evidence_sets(row: dict[str, Any]) -> list[tuple[str, ...]]:
    return sorted(tuple(sorted(str(value) for value in group)) for group in row.get("gold_evidence_sets", []))


def _retrieved_values(row: dict[str, Any], field: str) -> list[Any]:
    return [item.get(field) for item in row.get("retrieved", [])]


def _numeric_lists_match(left: list[Any], right: list[Any], tolerance: float) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if a is None or b is None:
            if a != b:
                return False
        elif not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance):
            return False
    return True


def compare_predictions(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    reference = _unique_by_question(reference_rows, Path("reference predictions"))
    candidate = _unique_by_question(candidate_rows, Path("candidate predictions"))
    common_ids = sorted(reference.keys() & candidate.keys())
    fields_match = {field: True for field in PREDICTION_FIELDS}
    mismatched_question_counts = {field: 0 for field in PREDICTION_FIELDS}

    for question_id in common_ids:
        left = reference[question_id]
        right = candidate[question_id]
        checks = {
            "ranked_parent_paragraph_ids_match": _retrieved_values(left, "paragraph_id")
            == _retrieved_values(right, "paragraph_id"),
            "ranked_chunk_ids_match": _retrieved_values(left, "chunk_id")
            == _retrieved_values(right, "chunk_id"),
            "ranked_text_match": _retrieved_values(left, "text")
            == _retrieved_values(right, "text"),
            "retrieval_scores_match": _numeric_lists_match(
                _retrieved_values(left, "score"),
                _retrieved_values(right, "score"),
                tolerance,
            ),
            "gold_evidence_sets_match": _evidence_sets(left) == _evidence_sets(right),
        }
        for field, matches in checks.items():
            fields_match[field] = fields_match[field] and matches
            if not matches:
                mismatched_question_counts[field] += 1

    same_question_ids = reference.keys() == candidate.keys()
    return {
        "reference_question_count": len(reference),
        "candidate_question_count": len(candidate),
        "shared_question_count": len(common_ids),
        "same_question_ids": same_question_ids,
        **fields_match,
        "mismatched_question_counts": mismatched_question_counts,
    }


def _read_metric_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = {}
        for row in reader:
            question_id = str(row["question_id"])
            if question_id in rows:
                raise ValueError(f"Duplicate question_id {question_id!r} in {path}")
            rows[question_id] = row
        return rows


def compare_metric_files(reference_path: Path, candidate_path: Path, tolerance: float) -> dict[str, Any]:
    if not reference_path.exists() or not candidate_path.exists():
        return {
            "available": False,
            "reference_exists": reference_path.exists(),
            "candidate_exists": candidate_path.exists(),
            "same_question_ids": None,
            "metrics_match": None,
        }
    reference = _read_metric_rows(reference_path)
    candidate = _read_metric_rows(candidate_path)
    same_ids = reference.keys() == candidate.keys()
    mismatched_rows = 0
    for question_id in reference.keys() & candidate.keys():
        left, right = reference[question_id], candidate[question_id]
        if any(
            not math.isclose(
                float(left[metric]), float(right[metric]), rel_tol=0.0, abs_tol=tolerance
            )
            for metric in METRICS
        ):
            mismatched_rows += 1
    return {
        "available": True,
        "reference_question_count": len(reference),
        "candidate_question_count": len(candidate),
        "same_question_ids": same_ids,
        "mismatched_metric_rows": mismatched_rows,
        "metrics_match": same_ids and mismatched_rows == 0,
        "reference_sha256": sha256_file(reference_path),
        "candidate_sha256": sha256_file(candidate_path),
    }


def _compare_provenance(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    if not reference_path.exists() or not candidate_path.exists():
        return {
            "available": False,
            "reference_exists": reference_path.exists(),
            "candidate_exists": candidate_path.exists(),
            "protocol_fields_match": None,
            "different_fields": [],
        }
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    differences = [
        key for key in PROVENANCE_FIELDS if reference.get(key) != candidate.get(key)
    ]
    return {
        "available": True,
        "fields_checked": list(PROVENANCE_FIELDS),
        "protocol_fields_match": not differences,
        "different_fields": differences,
        "reference_python_version": reference.get("python_version"),
        "candidate_python_version": candidate.get("python_version"),
        "reference_config_sha256": sha256_file(reference_path),
        "candidate_config_sha256": sha256_file(candidate_path),
    }


def compare_method_roots(
    label: str,
    reference_root: Path,
    candidate_root: Path,
    *,
    tolerance: float,
) -> dict[str, Any]:
    methods = {}
    for method in METHODS:
        reference_dir = reference_root / method / "evaluation" / "token-aware"
        candidate_dir = candidate_root / method / "evaluation" / "token-aware"
        reference_predictions = reference_dir / "predictions.jsonl"
        candidate_predictions = candidate_dir / "predictions.jsonl"
        reference_config = reference_dir / "experiment_config.json"
        candidate_config = candidate_dir / "experiment_config.json"
        for path in (reference_predictions, candidate_predictions):
            if not path.exists():
                raise FileNotFoundError(path)

        prediction_comparison = compare_predictions(
            read_jsonl(reference_predictions),
            read_jsonl(candidate_predictions),
            tolerance=tolerance,
        )
        methods[method] = {
            "predictions": {
                **prediction_comparison,
                "reference_sha256": sha256_file(reference_predictions),
                "candidate_sha256": sha256_file(candidate_predictions),
            },
            "retrieval_metrics": compare_metric_files(
                reference_dir / "retrieval_metrics.csv",
                candidate_dir / "retrieval_metrics.csv",
                tolerance,
            ),
            "provenance": _compare_provenance(reference_config, candidate_config),
        }
    return {"label": label, "reference_root": str(reference_root), "candidate_root": str(candidate_root), "methods": methods}


def compare_ablation_files(reference_path: Path, candidate_path: Path, tolerance: float) -> dict[str, Any]:
    key_fields = ("study", "method", "parameter_name", "parameter", "question_id")

    def read_rows(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = {}
            for row in csv.DictReader(stream):
                key = tuple(row[field] for field in key_fields)
                if key in rows:
                    raise ValueError(f"Duplicate ablation key {key!r} in {path}")
                rows[key] = row
            return rows

    reference = read_rows(reference_path)
    candidate = read_rows(candidate_path)
    shared = reference.keys() & candidate.keys()
    mismatched_rows = sum(
        any(
            not math.isclose(
                float(reference[key][metric]),
                float(candidate[key][metric]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            for metric in METRICS
        )
        for key in shared
    )
    same_keys = reference.keys() == candidate.keys()
    return {
        "reference_row_count": len(reference),
        "candidate_row_count": len(candidate),
        "same_question_study_keys": same_keys,
        "mismatched_metric_rows": mismatched_rows,
        "metrics_match": same_keys and mismatched_rows == 0,
        "reference_sha256": sha256_file(reference_path),
        "candidate_sha256": sha256_file(candidate_path),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    comparison_roots = (
        (
            "full_validation_k5",
            args.reference_full_root,
            args.python311_full_root,
        ),
        (
            "full_validation_k20",
            args.reference_full_root / "candidates20",
            args.python311_full_root / "candidates20",
        ),
        ("pilot_k5", args.reference_pilot_root, args.python311_pilot_root),
        (
            "pilot_k20",
            args.reference_pilot_root / "candidates20",
            args.python311_pilot_root / "candidates20",
        ),
    )
    comparisons = [
        compare_method_roots(label, reference, candidate, tolerance=args.tolerance)
        for label, reference, candidate in comparison_roots
    ]
    ablations = compare_ablation_files(
        args.reference_ablation_file, args.python311_ablation_file, args.tolerance
    )
    error_files = {}
    for label, reference_path, candidate_path in (
        (
            "full_validation",
            args.reference_full_root / "error_analysis.csv",
            args.python311_full_root / "error_analysis.csv",
        ),
        (
            "pilot",
            args.reference_pilot_error_file,
            args.python311_pilot_error_file,
        ),
    ):
        error_files[label] = {
            "reference_sha256": sha256_file(reference_path),
            "candidate_sha256": sha256_file(candidate_path),
            "byte_identical": reference_path.read_bytes() == candidate_path.read_bytes(),
        }
    all_prediction_checks_pass = all(
        method_result["predictions"]["same_question_ids"]
        and all(method_result["predictions"][field] for field in PREDICTION_FIELDS)
        and (
            not method_result["provenance"]["available"]
            or method_result["provenance"]["protocol_fields_match"]
        )
        and (
            not method_result["retrieval_metrics"]["available"]
            or method_result["retrieval_metrics"]["metrics_match"]
        )
        for comparison in comparisons
        for method_result in comparison["methods"].values()
    )
    errors_match = all(item["byte_identical"] for item in error_files.values())
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "tolerance": args.tolerance,
        "reference_python": "saved historical run configurations",
        "candidate_python": "saved Python 3.11 run configurations",
        "comparisons": comparisons,
        "development_ablations": ablations,
        "error_analysis_files": error_files,
        "all_predictions_and_provenance_match": all_prediction_checks_pass,
        "development_ablations_match": ablations["metrics_match"],
        "error_analysis_files_match": errors_match,
        "all_checks_pass": all_prediction_checks_pass and ablations["metrics_match"] and errors_match,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare saved Python 3.11 RAG outputs with the historical reference runs."
    )
    parser.add_argument("--reference-full-root", type=Path, default=Path("results/experiments/full_validation"))
    parser.add_argument("--python311-full-root", type=Path, default=Path("results/experiments/python311_full_validation"))
    parser.add_argument("--reference-pilot-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--python311-pilot-root", type=Path, default=Path("results/experiments/python311_pilot"))
    parser.add_argument("--reference-ablation-file", type=Path, default=Path("results/ablation_results.csv"))
    parser.add_argument(
        "--python311-ablation-file",
        type=Path,
        default=Path("results/experiments/python311_ablations/ablation_results.csv"),
    )
    parser.add_argument("--reference-pilot-error-file", type=Path, default=Path("results/error_analysis.csv"))
    parser.add_argument(
        "--python311-pilot-error-file",
        type=Path,
        default=Path("results/experiments/python311_pilot/error_analysis.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiments/python311_runtime_parity.json"),
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_checks_pass": report["all_checks_pass"]}, indent=2))
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
