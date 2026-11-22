"""Trace evidence availability, citation alignment, and abstention in saved runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from scientific_rag.evaluation import retrieval_metrics

if __package__:
    from .analyze_generation import unique_prediction_ids
else:
    from analyze_generation import unique_prediction_ids


def checked_rows(path: Path, expected_hash: str) -> list[dict]:
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_hash:
        raise ValueError(f"Input changed after scoring: {path}")
    return [json.loads(line) for line in contents.splitlines() if line.strip()]


def diagnose(
    method: str, question: dict, prediction: dict, score: dict,
    retrieval: dict | None, context_k: int,
) -> dict:
    annotations = question["annotations"]
    valid = [entry for entry in annotations if not entry["unanswerable"]]
    if not annotations:
        answerability = "unlabeled"
    elif not valid:
        answerability = "fully_unanswerable"
    elif len(valid) != len(annotations):
        answerability = "mixed_answerability"
    else:
        answerability = "answerable"

    gold = [set(entry["mapped_evidence_ids"]) for entry in valid
            if entry["mapped_evidence_ids"]]
    context_ids = [
        passage.get("paragraph_id") or passage["chunk_id"]
        for passage in retrieval["retrieved"][:context_k]
    ] if retrieval is not None else []
    citations = prediction["cited_paragraph_ids"]
    gold_union = set().union(*gold)
    gold_citations = set(citations) & gold_union
    recall = None
    if retrieval is None:
        evidence_status = "no_retrieval_context"
    elif not gold:
        evidence_status = "no_mapped_answerable_evidence"
    else:
        recall = retrieval_metrics(context_ids, gold, k=context_k)["evidence_recall"]
        if recall == 0:
            evidence_status = "no_mapped_evidence_in_context"
        elif recall == 1:
            evidence_status = "complete_annotator_set_in_context"
        else:
            evidence_status = "partial_annotator_set_in_context"

    return {
        "method": method, "question_id": question["question_id"],
        "paper_id": question["paper_id"], "question": question["question"],
        "answerability_label": answerability, "answer": prediction["answer"],
        "abstained": prediction["abstained"],
        "missed_abstention": answerability == "fully_unanswerable" and not prediction["abstained"],
        "incorrect_abstention": answerability == "answerable" and prediction["abstained"],
        "answer_exact_match": score["answer_exact_match"], "token_f1": score["token_f1"],
        "mapped_evidence_status": evidence_status, "evidence_recall_at_k": recall,
        "available_mapped_evidence_not_cited": bool(recall and not gold_citations),
        "zero_f1_with_mapped_citation": score["token_f1"] == 0 and bool(gold_citations),
        "zero_f1_with_complete_evidence": score["token_f1"] == 0 and recall == 1,
        "context_paragraph_ids": json.dumps(context_ids),
        "cited_paragraph_ids": json.dumps(citations),
        "mapped_citation_ids": json.dumps(sorted(gold_citations)),
        "mapped_evidence_sets": json.dumps([sorted(group) for group in gold]),
        "reference_answers_by_annotation": json.dumps(
            [entry["reference_answers"] for entry in valid], ensure_ascii=False,
        ),
    }


def summarize(rows: list[dict]) -> dict:
    mapped = [row for row in rows if row["evidence_recall_at_k"] is not None]
    return {
        "question_count": len(rows),
        "mapped_context_count": len(mapped),
        "mapped_evidence_status_counts": dict(Counter(
            row["mapped_evidence_status"] for row in mapped
        )),
        "answer_scored_count": sum(row["token_f1"] is not None for row in rows),
        "zero_f1_count": sum(row["token_f1"] == 0 for row in rows),
        "explicit_abstentions": sum(row["abstained"] for row in rows),
        **{field: sum(row[field] for row in rows) for field in (
            "missed_abstention", "incorrect_abstention", "available_mapped_evidence_not_cited",
            "zero_f1_with_mapped_citation", "zero_f1_with_complete_evidence",
        )},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contents = args.comparison.read_bytes()
    comparison = json.loads(contents)
    inputs = comparison["inputs"]
    questions = checked_rows(Path(inputs["question_file"]), inputs["question_file_sha256"])
    question_by_id = {row["question_id"]: row for row in questions}
    all_rows = []
    methods = {}
    for method, source in inputs["methods"].items():
        predictions = checked_rows(
            Path(source["prediction_file"]), source["prediction_file_sha256"],
        )
        scores = comparison["per_question_metrics"][method]
        ids = unique_prediction_ids(predictions, method)
        if ids != unique_prediction_ids(scores, f"{method} scores"):
            raise ValueError(f"Scored questions differ from predictions for {method}")
        scores_by_id = {row["question_id"]: row for row in scores}
        retrieval_by_id = {}
        if source["retrieval_file"]:
            contexts = checked_rows(
                Path(source["retrieval_file"]), source["retrieval_file_sha256"],
            )
            if ids - unique_prediction_ids(contexts, f"{method} retrieval"):
                raise ValueError(f"Missing contexts for {method}")
            retrieval_by_id = {row["question_id"]: row for row in contexts}
        context_k = comparison["method_metrics"][method]["citation_metrics"]["context_k"] or 5
        rows = []
        for prediction in predictions:
            qid = prediction["question_id"]
            rows.append(diagnose(
                method, question_by_id[qid], prediction, scores_by_id[qid],
                retrieval_by_id.get(qid), context_k,
            ))
        methods[method] = summarize(rows)
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("Comparison contains no predictions")
    summary = {
        "comparison_file": str(args.comparison),
        "comparison_sha256": hashlib.sha256(contents).hexdigest(),
        "case_rows": len(all_rows), "methods": methods,
        "interpretation": (
            "Counts describe the saved sample. Evidence and citations match paragraph IDs, "
            "and zero answer F1 means zero reference-token overlap. These flags do not "
            "establish whether individual claims are semantically supported."
        ),
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "generation_error_analysis.csv").open(
        "x", encoding="utf-8-sig", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / "generation_error_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
