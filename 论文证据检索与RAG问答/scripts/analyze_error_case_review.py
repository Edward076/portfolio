"""Summarize supplied human-review records, preserving pre-unblinding decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

if __package__:
    from .validate_error_case_review import FIELDS, read_decisions, validate_review
else:
    from validate_error_case_review import FIELDS, read_decisions, validate_review

LOCKED_FIELDS = tuple(field for field in FIELDS if field not in {
    "post_unblind_preference", "possible_annotation_gap", "reviewed_at_utc",
})
METHODS = {"hybrid", "hybrid_reranked"}


def resolve_preference(choice: str, mapping: dict) -> str:
    return mapping[f"method_for_context_{choice}"] if choice in {"A", "B"} else choice


def analyze_review(packet: Path, stage1: Path, decisions: Path, key: Path) -> dict:
    initial_check = validate_review(packet, stage1)
    if initial_check["stage1_status"] != "complete":
        raise ValueError("Stage-1 review is incomplete; finish it before unblinding analysis")
    initial = {row["question_id"]: row for row in read_decisions(stage1)}
    if any(row["post_unblind_preference"] or row["possible_annotation_gap"]
           for row in initial.values()):
        raise ValueError("Stage-1 snapshot already contains post-unblinding fields")
    final_check = validate_review(packet, decisions)
    if final_check["stage1_status"] != "complete":
        raise ValueError("Stage-2 records have incomplete stage-1 fields")
    final = {row["question_id"]: row for row in read_decisions(decisions)}
    for question_id, before in initial.items():
        after = final[question_id]
        if any(before[field] != after[field] for field in LOCKED_FIELDS):
            raise ValueError(f"{question_id}: stage-1 fields changed after unblinding")
        if not after["post_unblind_preference"] or not after["possible_annotation_gap"]:
            raise ValueError(f"{question_id}: missing stage-2 preference or annotation-gap choice")
        if datetime.fromisoformat(after["reviewed_at_utc"]) < datetime.fromisoformat(
            before["reviewed_at_utc"]
        ):
            raise ValueError(f"{question_id}: stage-2 review time precedes stage 1")

    # Read the explicitly supplied key only after the review records pass validation.
    key_rows = [json.loads(line) for line in key.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    keys = {row["question_id"]: row for row in key_rows}
    if len(keys) != len(key_rows) or keys.keys() != initial.keys():
        raise ValueError("Unblinding key must contain every packet question exactly once")
    for question_id, row in keys.items():
        if {row.get("method_for_context_A"), row.get("method_for_context_B")} != METHODS:
            raise ValueError(f"{question_id}: invalid A/B method mapping in key")
        if not isinstance(row.get("outcome_stratum"), str) or not row["outcome_stratum"].strip():
            raise ValueError(f"{question_id}: missing outcome stratum in key")

    pre, post, gaps = Counter(), Counter(), Counter()
    sufficient = {method: Counter() for method in sorted(METHODS)}
    strata = defaultdict(lambda: {"case_count": 0, "pre": Counter(), "post": Counter()})
    cases = []
    for question_id in sorted(initial):
        before, after, mapping = initial[question_id], final[question_id], keys[question_id]

        pre_choice = resolve_preference(before["pre_unblind_preference"], mapping)
        post_choice = resolve_preference(after["post_unblind_preference"], mapping)
        pre[pre_choice] += 1
        post[post_choice] += 1
        gaps[after["possible_annotation_gap"]] += 1
        stratum = mapping["outcome_stratum"]
        strata[stratum]["case_count"] += 1
        strata[stratum]["pre"][pre_choice] += 1
        strata[stratum]["post"][post_choice] += 1
        for label in ("A", "B"):
            sufficient[mapping[f"method_for_context_{label}"]][
                before[f"context_{label}_sufficiency"]
            ] += 1
        cases.append({"question_id": question_id, "outcome_stratum": stratum,
                      "pre_unblind_preference": pre_choice,
                      "post_unblind_preference": post_choice,
                      "preference_changed": pre_choice != post_choice})
    return {
        "scope": "Summary of supplied records; does not verify human identity or blinding",
        "case_count": len(initial), "reviewer_count": initial_check["reviewer_count"],
        "pre_unblind_preference_counts": dict(sorted(pre.items())),
        "post_unblind_preference_counts": dict(sorted(post.items())),
        "preference_changed_count": sum(row["preference_changed"] for row in cases),
        "stage1_sufficiency_by_method": sufficient,
        "stage2_possible_annotation_gap_counts": dict(sorted(gaps.items())),
        "by_outcome_stratum": dict(sorted(strata.items())), "cases": cases,
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in {"blinded_packet": packet, "stage1_csv": stage1,
                                            "stage2_csv": decisions, "unblinding_key": key}.items()},
        "limitations": (
            "One reviewer per file pair. Exploratory stratified sample, not benchmark-wide rates. "
            "The user must supply the original matching key and a preserved stage-1 snapshot; "
            "comparison cannot prove the snapshot predates key access. Multiple reviewers "
            "require a separate agreement analysis before any adjudication."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path(
        "results/experiments/error_case_review/review_packet_blinded.jsonl"))
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze_review(args.packet, args.stage1, args.decisions, args.key)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    report = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
