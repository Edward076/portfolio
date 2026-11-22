"""Offline answer and citation scoring for saved Qasper generation predictions."""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """Normalize an answer with an SQuAD-style lowercase/article/punctuation pass."""
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    if normalized in {"yes", "true"}:
        return "yes"
    if normalized in {"no", "false"}:
        return "no"
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return " ".join(token for token in tokens if token not in _ARTICLES)


def token_f1(prediction: str, reference: str) -> float:
    """Compute token F1 after answer normalization."""
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _valid_annotations(question: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [annotation for annotation in question["annotations"] if not annotation["unanswerable"]]


def _citation_ids(prediction: Mapping[str, Any]) -> set[str]:
    citations = prediction.get("cited_paragraph_ids", [])
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        raise TypeError("cited_paragraph_ids must be a list of paragraph ID strings")
    return set(citations)


def _context_ids(prediction: Mapping[str, Any], context_k: int) -> set[str]:
    retrieved = prediction.get("retrieved", [])[:context_k]
    return {
        item.get("paragraph_id") or item.get("chunk_id")
        for item in retrieved
        if item.get("paragraph_id") or item.get("chunk_id")
    }


def evaluate_generation(
    questions: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    retrieval_predictions: Iterable[Mapping[str, Any]] | None = None,
    context_k: int = 5,
) -> dict[str, Any]:
    """Score saved answers, abstentions, and paragraph citations against Qasper annotations.

    Answer EM/F1 use the best score across all answerable annotator references. Fully
    unanswerable questions are used for abstention classification; mixed-answerability
    questions are scored for answer quality but omitted from answerability accuracy.
    Citation recall uses the best answerable annotator evidence set, while citation
    precision uses the union across valid annotations. If retrieval outputs are supplied,
    groundedness checks citations against the first ``context_k`` retrieved paragraphs,
    regardless of answerability or whether gold evidence can be mapped to paragraphs.
    """
    if context_k <= 0:
        raise ValueError("context_k must be positive")

    question_rows = list(questions)
    question_by_id = {question["question_id"]: question for question in question_rows}
    if len(question_by_id) != len(question_rows):
        raise ValueError("Question records contain duplicate question_id values")

    prediction_by_id: dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        question_id = prediction.get("question_id")
        if not isinstance(question_id, str) or question_id not in question_by_id:
            raise ValueError(f"Prediction has an unknown or missing question_id: {question_id!r}")
        if question_id in prediction_by_id:
            raise ValueError(f"Duplicate prediction for question_id {question_id}")
        if not isinstance(prediction.get("answer"), str):
            raise TypeError(f"Prediction answer must be a string for question_id {question_id}")
        if not isinstance(prediction.get("abstained"), bool):
            raise TypeError(f"Prediction abstained must be a boolean for question_id {question_id}")
        _citation_ids(prediction)
        prediction_by_id[question_id] = prediction

    retrieval_by_id: dict[str, Mapping[str, Any]] = {}
    if retrieval_predictions is not None:
        for prediction in retrieval_predictions:
            question_id = prediction.get("question_id")
            if question_id not in question_by_id:
                raise ValueError(f"Retrieval record has an unknown question_id {question_id!r}")
            if question_id in retrieval_by_id:
                raise ValueError(f"Duplicate retrieval record for question_id {question_id}")
            retrieval_by_id[question_id] = prediction

    answer_em: list[float] = []
    answer_f1: list[float] = []
    answerability = Counter()
    citation_precision: list[float] = []
    citation_recall: list[float] = []
    citation_f1: list[float] = []
    citation_count = 0
    grounded_citation_count = 0
    groundedness_question_count = 0
    fully_unanswerable_count = 0
    mixed_answerability_count = 0
    citation_question_count = 0
    per_question_metrics: list[dict[str, Any]] = []

    for question_id, prediction in prediction_by_id.items():
        question = question_by_id[question_id]
        question_metrics: dict[str, Any] = {
            "question_id": question_id,
            "paper_id": question.get("paper_id"),
            "answer_exact_match": None,
            "token_f1": None,
            "answerability_correct": None,
            "citation_precision": None,
            "citation_recall": None,
            "citation_f1": None,
            "groundedness_per_cited_question": None,
            "grounded_citation_count": None,
            "citation_count": None,
        }
        annotations = question["annotations"]
        valid = _valid_annotations(question)
        references = [
            reference
            for annotation in valid
            for reference in annotation.get("reference_answers", [])
            if isinstance(reference, str) and reference.strip()
        ]

        if references:
            answer = prediction["answer"]
            exact_match = max(
                float(normalize_answer(answer) == normalize_answer(reference))
                for reference in references
            )
            f1 = max(token_f1(answer, reference) for reference in references)
            answer_em.append(exact_match)
            answer_f1.append(f1)
            question_metrics["answer_exact_match"] = exact_match
            question_metrics["token_f1"] = f1

        is_fully_unanswerable = bool(annotations) and not valid
        is_mixed = bool(valid) and len(valid) != len(annotations)
        if is_fully_unanswerable:
            fully_unanswerable_count += 1
        if is_mixed:
            mixed_answerability_count += 1
        else:
            answerability["n_questions"] += 1
            gold_answerable = bool(valid)
            predicted_answerable = not prediction["abstained"]
            question_metrics["answerability_correct"] = float(
                gold_answerable == predicted_answerable
            )
            if gold_answerable and predicted_answerable:
                answerability["true_positive"] += 1
            elif gold_answerable:
                answerability["false_negative"] += 1
            elif predicted_answerable:
                answerability["false_positive"] += 1
            else:
                answerability["true_negative"] += 1

        citations = _citation_ids(prediction)
        evidence_sets = [
            set(annotation.get("mapped_evidence_ids", []))
            for annotation in valid
            if annotation.get("mapped_evidence_ids")
        ]
        if evidence_sets:
            citation_question_count += 1
            gold_union = set().union(*evidence_sets)
            precision = len(citations & gold_union) / len(citations) if citations else 0.0
            recall = max(len(citations & gold) / len(gold) for gold in evidence_sets)
            citation_score_f1 = (
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
            citation_precision.append(precision)
            citation_recall.append(recall)
            citation_f1.append(citation_score_f1)
            question_metrics["citation_precision"] = precision
            question_metrics["citation_recall"] = recall
            question_metrics["citation_f1"] = citation_score_f1

        retrieval = retrieval_by_id.get(question_id)
        if retrieval is not None:
            groundedness_question_count += 1
            context = _context_ids(retrieval, context_k)
            citation_count += len(citations)
            grounded_for_question = len(citations & context)
            grounded_citation_count += grounded_for_question
            question_metrics["grounded_citation_count"] = grounded_for_question
            question_metrics["citation_count"] = len(citations)
            if citations:
                question_metrics["groundedness_per_cited_question"] = (
                    grounded_for_question / len(citations)
                )
        per_question_metrics.append(question_metrics)

    answerability_denominator = answerability["n_questions"]
    answerability_accuracy = (
        (answerability["true_positive"] + answerability["true_negative"])
        / answerability_denominator
        if answerability_denominator
        else None
    )
    citation_groundedness = grounded_citation_count / citation_count if citation_count else None

    return {
        "counts": {
            "question_count": len(question_rows),
            "prediction_count": len(prediction_by_id),
            "missing_prediction_count": len(question_rows) - len(prediction_by_id),
            "answer_scored_count": len(answer_f1),
            "fully_unanswerable_count": fully_unanswerable_count,
            "mixed_answerability_count": mixed_answerability_count,
            "answerability_scored_count": answerability_denominator,
            "citation_scored_count": citation_question_count,
        },
        "answer_metrics": {
            "exact_match_macro": _mean(answer_em),
            "token_f1_macro": _mean(answer_f1),
        },
        "answerability": {
            "accuracy": answerability_accuracy,
            **{
                key: answerability[key]
                for key in ("true_positive", "false_positive", "true_negative", "false_negative")
            },
            "mixed_answerability_excluded": True,
        },
        "citation_metrics": {
            "precision_macro": _mean(citation_precision),
            "recall_macro": _mean(citation_recall),
            "f1_macro": _mean(citation_f1),
            "groundedness_micro": citation_groundedness,
            "groundedness_question_count": groundedness_question_count,
            "citation_count_with_retrieval_context": citation_count,
            "citation_count_in_top_context": grounded_citation_count,
            "context_k": context_k if retrieval_predictions is not None else None,
        },
        "per_question": sorted(per_question_metrics, key=lambda row: row["question_id"]),
    }
