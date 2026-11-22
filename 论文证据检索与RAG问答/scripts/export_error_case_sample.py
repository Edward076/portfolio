from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

STRATA = (
    "reranker_recovered",
    "reranker_regressed",
    "candidate_generation_failure",
    "unresolved_ranking_failure",
)


def read_jsonl(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return {
            row[key]: row
            for line in stream
            if line.strip()
            for row in [json.loads(line)]
        }


def rank_of(gold: set[str], passages: list[dict[str, Any]]) -> str:
    ranks = [row["rank"] for row in passages if row["paragraph_id"] in gold]
    return ", ".join(map(str, ranks)) if ranks else "not in candidate list"


def excerpt(text: str, limit: int = 320) -> str:
    return " ".join(text.split())[:limit].rstrip() + ("…" if len(" ".join(text.split())) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a deterministic, stratified sample of full-validation retrieval cases."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--per-stratum", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/error_case_sample.md")
    )
    args = parser.parse_args()
    if args.per_stratum < 1:
        parser.error("--per-stratum must be at least 1")

    data_dir = args.root / "data/processed/full_validation"
    results_dir = args.root / "results/experiments/full_validation"
    questions = read_jsonl(data_dir / "evaluation_questions.jsonl")
    hybrid_top5 = read_jsonl(
        results_dir / "hybrid/evaluation/token-aware/predictions.jsonl"
    )
    hybrid_candidates20 = read_jsonl(
        results_dir / "candidates20/hybrid/evaluation/token-aware/predictions.jsonl"
    )
    reranked = read_jsonl(
        results_dir / "hybrid_reranked/evaluation/token-aware/predictions.jsonl"
    )
    chunks = read_jsonl(data_dir / "evaluation_chunks.jsonl", key="paragraph_id")

    grouped: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in STRATA}
    for question_id in sorted(
        hybrid_top5.keys() & hybrid_candidates20.keys() & reranked.keys() & questions.keys()
    ):
        base_top5 = hybrid_top5[question_id]
        base, rerank = hybrid_candidates20[question_id], reranked[question_id]
        gold = {
            paragraph_id
            for evidence_set in base["gold_evidence_sets"]
            for paragraph_id in evidence_set
        }
        base_candidates = base["retrieved"]
        rerank_candidates = rerank["retrieved"]
        in_candidate_set = bool(
            gold & {row["paragraph_id"] for row in base_candidates}
        )
        base_hit = bool(gold & {row["paragraph_id"] for row in base_top5["retrieved"]})
        rerank_hit = bool(
            gold & {row["paragraph_id"] for row in rerank_candidates[:5]}
        )
        if not base_hit and rerank_hit:
            stratum = "reranker_recovered"
        elif base_hit and not rerank_hit:
            stratum = "reranker_regressed"
        elif not in_candidate_set:
            stratum = "candidate_generation_failure"
        elif not base_hit and not rerank_hit:
            stratum = "unresolved_ranking_failure"
        else:
            continue
        grouped[stratum].append(
            {
                "question_id": question_id,
                "question": questions[question_id]["question"],
                "paper_id": base["paper_id"],
                "gold": gold,
                "hybrid": base_candidates,
                "reranked": rerank_candidates,
            }
        )

    selected: dict[str, list[dict[str, Any]]] = {}
    for stratum, cases in grouped.items():
        cases.sort(
            key=lambda case: hashlib.sha256(
                f"{args.seed}:{case['question_id']}".encode()
            ).hexdigest()
        )
        selected[stratum] = cases[: args.per_stratum]

    lines = [
        "# Stratified Full-Validation Error Case Sample",
        "",
        (
            "This file is a deterministic aid for case review, not an adjudicated human audit. "
            f"It samples up to {args.per_stratum} questions from each outcome stratum. "
            f"Selection sorts SHA-256 hashes of `{args.seed}:question_id`; regenerate with "
            "`python scripts/export_error_case_sample.py`."
        ),
        "",
        "| Outcome stratum | Available cases | Sampled |",
        "|---|---:|---:|",
    ]
    for stratum in STRATA:
        lines.append(
            f"| `{stratum}` | {len(grouped[stratum])} | {len(selected[stratum])} |"
        )

    labels = {
        "reranker_recovered": "Reranker recovered a top-five hit",
        "reranker_regressed": "Reranker changed a hit into a miss",
        "candidate_generation_failure": "Hybrid candidate-generation failure",
        "unresolved_ranking_failure": "Evidence candidate remains below top five",
    }
    for stratum in STRATA:
        lines.extend(["", f"## {labels[stratum]}", ""])
        for case in selected[stratum]:
            gold = case["gold"]
            lines.extend(
                [
                    f"### {case['question_id']} — paper `{case['paper_id']}`",
                    "",
                    f"**Question:** {case['question']}",
                    "",
                    f"**Annotated evidence paragraph(s):** {', '.join(f'`{pid}`' for pid in sorted(gold))}",
                    "",
                    (
                        f"**Gold ranks:** Hybrid: {rank_of(gold, case['hybrid'])}; "
                        f"Hybrid + reranker: {rank_of(gold, case['reranked'])}."
                    ),
                    "",
                ]
            )
            for paragraph_id in sorted(gold):
                evidence = chunks.get(paragraph_id)
                lines.append(
                    f"**Evidence excerpt (`{paragraph_id}`):** "
                    + (excerpt(evidence["text"]) if evidence else "source passage unavailable")
                )
                lines.append("")
            lines.append("---")

    output = args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "seed": args.seed,
                "per_stratum": args.per_stratum,
                "available": {key: len(value) for key, value in grouped.items()},
                "sampled": {key: len(value) for key, value in selected.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
