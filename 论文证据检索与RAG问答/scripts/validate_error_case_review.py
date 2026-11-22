"""Validate recorded stage-1 review fields without reading the unblinding key."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

FIELDS = (
    "question_id", "context_A_sufficiency", "context_A_relevant_passage_ids",
    "context_B_sufficiency", "context_B_relevant_passage_ids", "pre_unblind_preference",
    "confidence_1_to_3", "post_unblind_preference", "possible_annotation_gap",
    "reviewer_notes", "reviewer_id", "reviewed_at_utc",
)
SUFFICIENCY = {"sufficient", "partly_sufficient", "insufficient", "uncertain"}
PREFERENCE = {"A", "B", "tie", "neither"}
GAPS = {
    "none", "possible_missing_evidence", "unmapped_evidence", "ambiguous_question",
    "passage_fragmentation", "other",
}
REQUIRED = (
    "context_A_sufficiency", "context_B_sufficiency", "pre_unblind_preference",
    "confidence_1_to_3", "reviewer_id", "reviewed_at_utc",
)


def read_decisions(decisions_path: Path) -> list[dict[str, str]]:
    with decisions_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or set(headers) != set(FIELDS):
            raise ValueError("CSV must contain exactly the documented columns, without duplicates")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV row has a missing or extra column")
    return [{key: value.strip() for key, value in row.items()} for row in rows]


def validate_review(packet_path: Path, decisions_path: Path) -> dict:
    packets = [json.loads(line) for line in packet_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    packet_by_id = {row["question_id"]: row for row in packets}
    if not packets or len(packet_by_id) != len(packets):
        raise ValueError("Packet must contain nonempty, unique question IDs")
    if any(not isinstance(key, str) or not key.strip() for key in packet_by_id):
        raise ValueError("Packet has an invalid question ID")
    rows = read_decisions(decisions_path)
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(packet_by_id):
        raise ValueError("CSV must contain every packet question exactly once")
    reviewers = {row["reviewer_id"] for row in rows if row["reviewer_id"]}
    if len(reviewers) > 1:
        raise ValueError("Use one CSV file per reviewer")

    statuses = []
    preferences: Counter = Counter()
    for row in rows:
        question_id = row["question_id"]
        choices = {
            "context_A_sufficiency": SUFFICIENCY, "context_B_sufficiency": SUFFICIENCY,
            "pre_unblind_preference": PREFERENCE, "post_unblind_preference": PREFERENCE,
            "confidence_1_to_3": {"1", "2", "3"}, "possible_annotation_gap": GAPS,
        }
        for field, allowed in choices.items():
            if row[field] and row[field] not in allowed:
                raise ValueError(f"{question_id}: invalid {field}: {row[field]!r}")
        timestamp = row["reviewed_at_utc"]
        if timestamp:
            try:
                parsed = datetime.fromisoformat(timestamp)
            except ValueError as error:
                raise ValueError(f"{question_id}: invalid reviewed_at_utc") from error
            if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
                raise ValueError(f"{question_id}: reviewed_at_utc must include UTC timezone")
        required = list(REQUIRED)
        for label in ("A", "B"):
            field = f"context_{label}_relevant_passage_ids"
            values = row[field].split(";") if row[field] else []
            values = [value.strip() for value in values]
            valid_ids = {item["chunk_id"] for item in packet_by_id[question_id][f"context_{label}"]}
            if len(values) != len(set(values)) or any(value not in valid_ids for value in values):
                raise ValueError(f"{question_id}: {field} must use unique chunk IDs from its context")
            sufficiency = row[f"context_{label}_sufficiency"]
            if sufficiency in {"sufficient", "partly_sufficient"}:
                required.append(field)
            if sufficiency == "uncertain":
                required.append("reviewer_notes")
        missing = sorted({field for field in required if not row[field]})
        if not any(row[field] for field in FIELDS if field != "question_id"):
            status = "blank"
        elif missing:
            status = "partial"
        else:
            status = "complete"
            preferences[row["pre_unblind_preference"]] += 1
        statuses.append({"question_id": question_id, "stage1_status": status,
                         "missing_stage1_fields": missing})
    counts = Counter(item["stage1_status"] for item in statuses)
    complete = counts["complete"]
    return {
        "scope": "Recorded stage-1 field completeness only; not verification of human review",
        "stage1_status": "complete" if complete == len(rows) else (
            "pending" if counts["blank"] == len(rows) else "partial"
        ),
        "case_count": len(rows), "complete_stage1_count": complete,
        "partial_stage1_count": counts["partial"], "blank_count": counts["blank"],
        "reviewer_count": len(reviewers),
        "pre_unblind_preference_counts_completed_rows": dict(sorted(preferences.items())),
        "stage2_review": "not assessed; no unblinding key was read",
        "cases": statuses,
        "input_sha256": {
            "blinded_packet": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            "decisions_csv": hashlib.sha256(decisions_path.read_bytes()).hexdigest(),
        },
        "interpretation": "Exploratory case sample; counts do not estimate benchmark-wide rates",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path(
        "results/experiments/error_case_review/review_packet_blinded.jsonl"))
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="optional new report file; never overwrites")
    args = parser.parse_args()
    try:
        result = validate_review(args.packet, args.decisions)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    report = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
