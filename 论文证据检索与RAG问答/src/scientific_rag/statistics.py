from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: str,
    cluster_key: str = "paper_id",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Percentile CI for a mean, resampling papers and retaining their questions."""
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            clusters[str(row[cluster_key])].append(float(value))
    if not clusters:
        return {"n_questions": 0, "n_clusters": 0, "mean": None, "ci_low": None, "ci_high": None}
    cluster_ids = sorted(clusters)
    observed = [value for values in clusters.values() for value in values]
    mean = sum(observed) / len(observed)
    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [value for cluster_id in sampled for value in clusters[cluster_id]]
        bootstrap_means.append(sum(values) / len(values))
    bootstrap_means.sort()
    low_index = int(0.025 * (n_bootstrap - 1))
    high_index = int(0.975 * (n_bootstrap - 1))
    return {
        "n_questions": len(observed),
        "n_clusters": len(cluster_ids),
        "mean": mean,
        "ci_low": bootstrap_means[low_index],
        "ci_high": bootstrap_means[high_index],
    }


def paired_cluster_bootstrap_ci(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | None]:
    left_by_question = {row["question_id"]: row for row in left}
    right_by_question = {row["question_id"]: row for row in right}
    differences: dict[str, list[float]] = defaultdict(list)
    for question_id in sorted(left_by_question.keys() & right_by_question.keys()):
        left_value = left_by_question[question_id].get(metric)
        right_value = right_by_question[question_id].get(metric)
        if left_value is None or right_value is None:
            continue
        paper_id = str(left_by_question[question_id]["paper_id"])
        differences[paper_id].append(float(right_value) - float(left_value))
    if not differences:
        return {
            "n_questions": 0,
            "n_clusters": 0,
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
        }
    cluster_ids = sorted(differences)
    all_differences = [value for values in differences.values() for value in values]
    mean = sum(all_differences) / len(all_differences)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [value for cluster_id in sampled for value in differences[cluster_id]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return {
        "n_questions": len(all_differences),
        "n_clusters": len(cluster_ids),
        "mean_difference": mean,
        "ci_low": draws[int(0.025 * (n_bootstrap - 1))],
        "ci_high": draws[int(0.975 * (n_bootstrap - 1))],
    }
