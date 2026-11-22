from __future__ import annotations

import time
from typing import Any


def retrieval_metrics(
    retrieved_ids: list[str], evidence_sets: list[set[str]], k: int = 5
) -> dict[str, float | int | None]:
    """Compute evidence metrics using the best alternative annotator set.

    Annotations are alternatives, so complete coverage means recovering all
    mapped evidence in at least one answer annotation, not the union of every
    annotator's potentially divergent evidence.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    usable_sets = [set(ids) for ids in evidence_sets if ids]
    if not usable_sets:
        return {
            "evidence_recall": None,
            "mrr": None,
            "hit_rate": None,
            "complete_evidence_coverage": None,
        }
    top_ids = retrieved_ids[:k]
    rank_by_id: dict[str, int] = {}
    for rank, chunk_id in enumerate(top_ids, start=1):
        rank_by_id.setdefault(chunk_id, rank)
    relevant_union = set().union(*usable_sets)
    first_relevant_rank = min(
        (rank_by_id[item] for item in relevant_union if item in rank_by_id), default=None
    )
    recall = max(len(gold & set(top_ids)) / len(gold) for gold in usable_sets)
    complete = max(float(gold.issubset(set(top_ids))) for gold in usable_sets)
    return {
        "evidence_recall": recall,
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "hit_rate": float(first_relevant_rank is not None),
        "complete_evidence_coverage": complete,
    }


def evaluate_retrieval(
    questions: list[dict[str, Any]],
    retriever,
    top_k: int = 5,
    include_unscored_predictions: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    method_name = getattr(retriever, "method_name", "unknown")
    for question in questions:
        answerable = [a for a in question["annotations"] if not a["unanswerable"]]
        evidence_sets = [set(a["mapped_evidence_ids"]) for a in answerable]
        scoreable = bool(answerable) and any(evidence_sets)
        if not scoreable and not include_unscored_predictions:
            continue
        started = time.perf_counter()
        hits = retriever.retrieve(question["question"], question["paper_id"], top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        component_latencies = getattr(retriever, "last_component_latency_ms", {})
        retrieved_ids = [row.get("paragraph_id", row["chunk_id"]) for row in hits]
        metrics = retrieval_metrics(retrieved_ids, evidence_sets, k=top_k)
        if scoreable:
            rows.append(
                {
                    "question_id": question["question_id"],
                    "paper_id": question["paper_id"],
                    "retriever": method_name,
                    "top_k": top_k,
                    "retrieval_latency_ms": latency_ms,
                    **component_latencies,
                    **metrics,
                }
            )
        predictions.append(
            {
                "question_id": question["question_id"],
                "paper_id": question["paper_id"],
                "method": method_name,
                "retrieval_latency_ms": latency_ms,
                **component_latencies,
                "retrieved": hits,
                "gold_evidence_sets": [sorted(gold) for gold in evidence_sets if gold],
            }
        )
    return rows, predictions


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("evidence_recall", "mrr", "hit_rate", "complete_evidence_coverage")
    return {
        "method": rows[0]["retriever"] if rows else None,
        "question_count": len(rows),
        **{
            key: sum(float(row[key]) for row in rows if row[key] is not None)
            / sum(row[key] is not None for row in rows)
            if any(row[key] is not None for row in rows)
            else None
            for key in keys
        },
    }
