from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HEADINGS = {
    "Reranker recovered a top-five hit": "reranker_recovered",
    "Reranker changed a hit into a miss": "reranker_regressed",
    "Hybrid candidate-generation failure": "candidate_generation_failure",
    "Evidence candidate remains below top five": "unresolved_ranking_failure",
}
CONTEXT_K = 5


def read_jsonl(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} values in {path}")
    return result


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sample(path: Path) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    stratum: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line[3:]
            if heading not in HEADINGS:
                raise ValueError(f"Unknown outcome heading in sample: {heading}")
            stratum = HEADINGS[heading]
            continue
        match = re.match(r"^### ([0-9a-f]{40}) — paper `([^`]+)`$", line)
        if match:
            if stratum is None:
                raise ValueError("Found a case before its outcome heading")
            cases.append((match.group(1), stratum))
    if not cases:
        raise ValueError(f"No question cases found in {path}")
    if len({question_id for question_id, _ in cases}) != len(cases):
        raise ValueError(f"Duplicate question IDs in {path}")
    return cases


def _context_passages(prediction: dict[str, Any]) -> list[dict[str, str]]:
    passages = prediction.get("retrieved", [])[:CONTEXT_K]
    return [
        {
            "chunk_id": passage.get("chunk_id") or passage["paragraph_id"],
            "paragraph_id": passage["paragraph_id"],
            "section_name": passage.get("section_name") or "",
            "text": passage["text"],
        }
        for passage in passages
    ]


def build_case_records(
    question: dict[str, Any],
    hybrid: dict[str, Any],
    reranked: dict[str, Any],
    stratum: str,
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    question_id = question["question_id"]
    if hybrid["question_id"] != question_id or reranked["question_id"] != question_id:
        raise ValueError(f"Question ID mismatch for {question_id}")
    method_for_a, method_for_b = ("hybrid", "hybrid_reranked")
    if hashlib.sha256(f"{seed}:{question_id}:blind-presentation".encode()).digest()[0] % 2:
        method_for_a, method_for_b = method_for_b, method_for_a
    method_predictions = {"hybrid": hybrid, "hybrid_reranked": reranked}
    blinded = {
        "question_id": question_id,
        "paper_id": question["paper_id"],
        "paper_title": question["paper_title"],
        "question": question["question"],
        "context_A": _context_passages(method_predictions[method_for_a]),
        "context_B": _context_passages(method_predictions[method_for_b]),
    }
    if not blinded["context_A"] or not blinded["context_B"]:
        raise ValueError(f"Missing retrieved context for {question_id}")
    key = {
        "question_id": question_id,
        "outcome_stratum": stratum,
        "method_for_context_A": method_for_a,
        "method_for_context_B": method_for_b,
        "mapped_gold_evidence_sets": hybrid.get("gold_evidence_sets", []),
        "annotations": [
            {
                "unanswerable": annotation.get("unanswerable"),
                "reference_answers": annotation.get("reference_answers", []),
                "mapped_evidence_ids": annotation.get("mapped_evidence_ids", []),
                "unmapped_evidence_texts": annotation.get("unmapped_evidence_texts", []),
            }
            for annotation in question.get("annotations", [])
        ],
    }
    return blinded, key


def make_decision_row(question_id: str) -> dict[str, str]:
    return {
        "question_id": question_id,
        "context_A_sufficiency": "",
        "context_A_relevant_passage_ids": "",
        "context_B_sufficiency": "",
        "context_B_relevant_passage_ids": "",
        "pre_unblind_preference": "",
        "confidence_1_to_3": "",
        "post_unblind_preference": "",
        "possible_annotation_gap": "",
        "reviewer_notes": "",
        "reviewer_id": "",
        "reviewed_at_utc": "",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def render_blinded_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Method-Blinded Retrieval Review Packet",
        "",
        "Review context A and B without opening `review_key.jsonl`. Record judgments in a copy of the CSV template before unblinding.",
        "",
        f"Cases: {len(rows)}. Each context shows its retrieved top {CONTEXT_K} passages in rank order.",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                "",
                "---",
                "",
                f"## Case {index:02d} — {row['question_id']}",
                "",
                f"**Paper:** {row['paper_title']} (`{row['paper_id']}`)",
                "",
                f"**Question:** {row['question']}",
            ]
        )
        for context_label in ("A", "B"):
            lines.extend(["", f"### Context {context_label}", ""])
            for passage_index, passage in enumerate(row[f"context_{context_label}"], start=1):
                lines.extend(
                    [
                        (
                            f"#### Passage {passage_index}: chunk "
                            f"`{passage['chunk_id']}` (parent paragraph `{passage['paragraph_id']}`)"
                        ),
                        "",
                    ]
                )
                if passage["section_name"]:
                    lines.extend([f"Section: {passage['section_name']}", ""])
                lines.extend([passage["text"], ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a method-blinded review packet for sampled retrieval cases."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", default="42")
    parser.add_argument(
        "--sample", type=Path, default=Path("docs/error_case_sample.md")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/experiments/error_case_review"),
        help="unused directory; existing directories are never overwritten",
    )
    args = parser.parse_args()

    root = args.root
    output_dir = root / args.output_dir
    if output_dir.exists():
        parser.error(
            f"Output directory already exists: {output_dir}. "
            "Use a new --output-dir to preserve review packets and human judgments."
        )
    data_dir = root / "data/processed/full_validation"
    results_dir = root / "results/experiments/full_validation"
    sample_path = root / args.sample
    question_path = data_dir / "evaluation_questions.jsonl"
    hybrid_path = results_dir / "hybrid/evaluation/token-aware/predictions.jsonl"
    reranked_path = results_dir / "hybrid_reranked/evaluation/token-aware/predictions.jsonl"
    questions = read_jsonl(question_path)
    hybrid = read_jsonl(hybrid_path)
    reranked = read_jsonl(reranked_path)

    blinded_rows = []
    key_rows = []
    for question_id, stratum in read_sample(sample_path):
        if question_id not in questions or question_id not in hybrid or question_id not in reranked:
            raise ValueError(f"Sample case {question_id} is absent from an input file")
        blinded, key = build_case_records(
            questions[question_id], hybrid[question_id], reranked[question_id], stratum, args.seed
        )
        blinded_rows.append(blinded)
        key_rows.append(key)

    # Exclusive creation also prevents a concurrent preparation from overwriting files.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "review_packet_blinded.jsonl", blinded_rows)
    (output_dir / "review_packet_blinded.md").write_text(
        render_blinded_markdown(blinded_rows), encoding="utf-8"
    )
    write_jsonl(output_dir / "review_key.jsonl", key_rows)
    with (output_dir / "review_decisions_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = [make_decision_row(row["question_id"]) for row in blinded_rows]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "context_k": CONTEXT_K,
        "case_count": len(blinded_rows),
        "blinded_packet": "review_packet_blinded.jsonl",
        "readable_blinded_packet": "review_packet_blinded.md",
        "unblinding_key": "review_key.jsonl",
        "decision_template": "review_decisions_template.csv",
        "input_sha256": {
            "sample_markdown": sha256_file(sample_path),
            "questions": sha256_file(question_path),
            "hybrid_predictions": sha256_file(hybrid_path),
            "reranked_predictions": sha256_file(reranked_path),
        },
        "description": (
            "A/B contexts are method-blinded with passage order preserved. The separate key "
            "contains outcome strata, method mappings, and Qasper reference annotations."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
