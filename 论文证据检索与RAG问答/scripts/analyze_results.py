from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from scientific_rag.statistics import cluster_bootstrap_ci, paired_cluster_bootstrap_ci

METRICS = ("evidence_recall", "mrr", "hit_rate", "complete_evidence_coverage")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def labels(question: dict) -> set[str]:
    annotations = question["annotations"]
    valid = [annotation for annotation in annotations if not annotation["unanswerable"]]
    if not valid:
        return {"unanswerable"}
    result = set()
    if any(annotation["yes_no"] is not None for annotation in valid):
        result.add("yes_no")
    if any(annotation["extractive_spans"] for annotation in valid):
        result.add("extractive")
    if any(annotation["free_form_answer"] for annotation in valid):
        result.add("free_form")
    if {annotation["unanswerable"] for annotation in annotations} == {True, False}:
        result.add("mixed_answerability")
    return result or {"other_answerable"}


def summarize(frame: pd.DataFrame, n_bootstrap: int, question_file: Path) -> dict:
    if "chunking" not in frame:
        frame = frame.assign(chunking="paragraph")
    method_groups = {
        (method, chunking): group.to_dict("records")
        for (method, chunking), group in frame.groupby(["retriever", "chunking"])
    }
    method_key = lambda method, chunking: f"{method} ({chunking})"
    output: dict = {"methods": {}, "paired_differences_vs_bm25": {}}
    for (method, chunking), rows in method_groups.items():
        output["methods"][method_key(method, chunking)] = {
            metric: cluster_bootstrap_ci(rows, metric, n_bootstrap=n_bootstrap)
            for metric in METRICS
        }
    for (method, chunking), rows in method_groups.items():
        baseline = method_groups.get(("bm25", chunking))
        if baseline is not None and method != "bm25":
            output["paired_differences_vs_bm25"][method_key(method, chunking)] = {
                metric: paired_cluster_bootstrap_ci(baseline, rows, metric, n_bootstrap=n_bootstrap)
                for metric in METRICS
            }
    output["paired_differences_between_methods"] = {}
    grouped_by_chunking: dict[str, list[tuple[str, list[dict]]]] = {}
    for (method, chunking), rows in method_groups.items():
        grouped_by_chunking.setdefault(chunking, []).append((method, rows))
    for chunking, entries in grouped_by_chunking.items():
        for left_index, (left_method, left_rows) in enumerate(entries):
            for right_method, right_rows in entries[left_index + 1 :]:
                pair_name = (
                    f"{method_key(left_method, chunking)} -> {method_key(right_method, chunking)}"
                )
                output["paired_differences_between_methods"][pair_name] = {
                    metric: paired_cluster_bootstrap_ci(
                        left_rows, right_rows, metric, n_bootstrap=n_bootstrap
                    )
                    for metric in METRICS
                }
    output["by_question_type_nonexclusive"] = {}
    if question_file.exists():
        question_labels = {
            question["question_id"]: labels(question) for question in read_jsonl(question_file)
        }
        frame = frame.copy()
        frame["question_types"] = frame["question_id"].map(question_labels)
        for question_type in ("extractive", "free_form", "yes_no", "mixed_answerability"):
            subset = frame[
                frame["question_types"].map(lambda types, target=question_type: target in types)
            ]
            output["by_question_type_nonexclusive"][question_type] = {
                method_key(method, chunking): {
                    metric: cluster_bootstrap_ci(
                        group.to_dict("records"), metric, n_bootstrap=n_bootstrap
                    )
                    for metric in METRICS
                }
                for (method, chunking), group in subset.groupby(["retriever", "chunking"])
            }
    output["bootstrap"] = {
        "unit": "paper_id",
        "replicates": n_bootstrap,
        "seed": 42,
        "interval": "percentile 95%",
    }
    return output


def summarize_latency(results_dir: Path) -> dict:
    summaries = {}
    for path in results_dir.rglob("latency_metrics.csv"):
        if "evaluation" not in path.parts or "candidates20" in path.parts:
            continue
        frame = pd.read_csv(path)
        if frame.empty or "retrieval_latency_ms" not in frame:
            continue
        chunking = (
            path.parent.name if path.parent.name in {"paragraph", "token-aware"} else "paragraph"
        )
        method = frame["method"].iloc[0]
        values = sorted(float(value) for value in frame["retrieval_latency_ms"].dropna())
        if not values:
            continue
        summaries[f"{method} ({chunking})"] = {
            "question_count": len(values),
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p95_ms": values[min(int(len(values) * 0.95), len(values) - 1)],
            "components_mean_ms": {
                column: statistics.fmean(float(value) for value in frame[column].dropna())
                for column in frame.columns
                if column.endswith("latency_ms") and column != "retrieval_latency_ms"
            },
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-bootstrap retrieval metrics by paper.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/experiments"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiments/retrieval_statistical_summary.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--question-file", type=Path, default=Path("data/processed/evaluation_questions.jsonl")
    )
    args = parser.parse_args()
    metric_files = [
        path
        for path in args.results_dir.rglob("retrieval_metrics.csv")
        if "evaluation" in path.parts and "candidates20" not in path.parts
    ]
    frames = [pd.read_csv(path) for path in metric_files]
    if not frames:
        raise SystemExit("No retrieval metrics found. Run retrieval experiments first.")
    frame = pd.concat(frames, ignore_index=True)
    summary = summarize(frame, args.bootstrap_replicates, args.question_file)
    summary["latency_summary"] = summarize_latency(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Wrote cluster-bootstrap summary for {frame['retriever'].nunique()} method(s) to {args.output}"
    )


if __name__ == "__main__":
    main()
