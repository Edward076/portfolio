"""Diagnose unmapped Qasper evidence without changing the exact-mapping labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scientific_rag.preprocessing import clean_text, evidence_key

SAMPLE_SEED = "42"
MIN_SUBSTRING_LENGTH = 60
DIAGNOSTIC_CATEGORIES = (
    "marker_stripped_full_text_candidate",
    "punctuation_or_symbol_only_full_text_candidate",
    "unique_long_substring_candidate",
    "multiple_long_substring_candidates",
    "no_compact_candidate_found",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_key(text: str) -> str:
    """Remove punctuation/spacing for diagnostic candidates only, never official mapping."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def classify_unmapped_evidence(
    evidence_text: str, paragraphs: list[dict[str, str]]
) -> dict[str, Any]:
    """Find same-paper full-text/substring candidates; report candidates as unverified."""
    evidence_clean = clean_text(evidence_text)
    has_float_marker = bool(
        re.match(r"^FLOAT\s+SELECTED\s*:", evidence_clean, flags=re.IGNORECASE)
    )
    comparison_text = re.sub(
        r"^FLOAT\s+SELECTED\s*:\s*", "", evidence_clean, flags=re.IGNORECASE
    )
    compact_evidence = compact_key(comparison_text)
    compact_paragraphs = [
        (paragraph["paragraph_id"], compact_key(paragraph["text"]))
        for paragraph in paragraphs
    ]
    full_matches = [
        paragraph_id
        for paragraph_id, compact_paragraph in compact_paragraphs
        if compact_evidence and compact_evidence == compact_paragraph
    ]
    if full_matches:
        category = (
            "marker_stripped_full_text_candidate"
            if has_float_marker
            else "punctuation_or_symbol_only_full_text_candidate"
        )
        candidates = full_matches
    elif len(compact_evidence) >= MIN_SUBSTRING_LENGTH:
        substring_matches = [
            paragraph_id
            for paragraph_id, compact_paragraph in compact_paragraphs
            if len(compact_paragraph) >= MIN_SUBSTRING_LENGTH
            and (compact_evidence in compact_paragraph or compact_paragraph in compact_evidence)
        ]
        if len(substring_matches) == 1:
            category = "unique_long_substring_candidate"
        elif substring_matches:
            category = "multiple_long_substring_candidates"
        else:
            category = "no_compact_candidate_found"
        candidates = substring_matches
    else:
        category = "no_compact_candidate_found"
        candidates = []
    return {
        "diagnostic_category": category,
        "candidate_paragraph_ids": candidates,
        "has_float_selected_marker": has_float_marker,
        "compact_length": len(compact_evidence),
    }


def collect_unmapped_items(
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    captions: list[dict[str, Any]] | None = None,
    sections: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    paragraphs_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    sections_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sections_by_paper_and_key: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        paragraphs_by_paper[str(chunk["paper_id"])].append(
            {"paragraph_id": chunk["paragraph_id"], "text": chunk["text"]}
        )
        section_name = clean_text(chunk.get("section_name"))
        paper_id = str(chunk["paper_id"])
        if section_name and evidence_key(section_name) not in sections_by_paper_and_key[paper_id]:
            sections_by_paper[paper_id].append(
                {"section_id": "", "text": section_name, "nonempty_paragraph_count": None}
            )
            sections_by_paper_and_key[paper_id].add(evidence_key(section_name))
    if sections is not None:
        sections_by_paper.clear()
        for section in sections:
            sections_by_paper[str(section["paper_id"])].append(section)
    captions_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for caption in captions or []:
        captions_by_paper[str(caption["paper_id"])].append(caption)
    items = []
    for question in questions:
        paper_id = str(question["paper_id"])
        paper_paragraphs = paragraphs_by_paper.get(paper_id, [])
        paper_captions = captions_by_paper.get(paper_id, [])
        for annotation in question["annotations"]:
            for evidence_text in annotation.get("unmapped_evidence_texts", []):
                diagnostic = classify_unmapped_evidence(evidence_text, paper_paragraphs)
                clean_evidence = clean_text(evidence_text)
                comparison_evidence = re.sub(
                    r"^FLOAT\s+SELECTED\s*:\s*", "", clean_evidence, flags=re.IGNORECASE
                )
                caption_matches = [
                    caption["caption_id"]
                    for caption in paper_captions
                    if evidence_key(comparison_evidence) == evidence_key(caption["text"])
                ]
                matching_sections = [
                    section
                    for section in sections_by_paper.get(paper_id, [])
                    if evidence_key(clean_evidence) == evidence_key(section["text"])
                ]
                section_matches = sorted({section["text"] for section in matching_sections})
                section_matches_without_body = [
                    section.get("section_id", "")
                    for section in matching_sections
                    if section.get("nonempty_paragraph_count") == 0
                ]
                items.append(
                    {
                        "paper_id": paper_id,
                        "question_id": question["question_id"],
                        "annotation_index": annotation["annotation_index"],
                        "evidence_text": clean_text(evidence_text),
                        "evidence_key": evidence_key(evidence_text),
                        "caption_match_ids": caption_matches,
                        "has_caption_match": bool(caption_matches),
                        "section_heading_matches": sorted(section_matches),
                        "section_heading_match_ids": sorted(
                            section.get("section_id", "") for section in matching_sections
                        ),
                        "section_heading_matches_without_body": sorted(
                            section_matches_without_body
                        ),
                        "has_section_heading_match": bool(section_matches),
                        **diagnostic,
                    }
                )
    return items


def _excerpt(text: str, limit: int = 360) -> str:
    return text[:limit].rstrip() + ("…" if len(text) > limit else "")


def select_examples(items: list[dict[str, Any]], per_category: int = 3) -> dict[str, list[dict[str, Any]]]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        categories[item["diagnostic_category"]].append(item)
    selected = {}
    for category, rows in categories.items():
        rows.sort(
            key=lambda item: hashlib.sha256(
                f"{SAMPLE_SEED}:{item['question_id']}:{item['annotation_index']}:{item['evidence_key']}".encode()
            ).hexdigest()
        )
        selected[category] = rows[:per_category]
    return dict(sorted(selected.items()))


def render_report(
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    captions: list[dict[str, Any]],
    items: list[dict[str, Any]],
    sections: list[dict[str, Any]] | None = None,
) -> str:
    total_evidence_items = sum(
        len(annotation.get("evidence_texts", []))
        for question in questions
        for annotation in question["annotations"]
    )
    unique_texts = {item["evidence_key"] for item in items}
    category_counts = Counter(item["diagnostic_category"] for item in items)
    marked_by_category = Counter(
        item["diagnostic_category"]
        for item in items
        if item["has_float_selected_marker"]
    )
    marker_count = sum(item["has_float_selected_marker"] for item in items)
    caption_matches = sum(item["has_caption_match"] for item in items)
    section_matches = sum(item["has_section_heading_match"] for item in items)
    section_matches_without_body = sum(
        bool(item["section_heading_matches_without_body"]) for item in items
    )
    caption_and_section = sum(
        item["has_caption_match"] and item["has_section_heading_match"] for item in items
    )
    caption_only = sum(
        item["has_caption_match"] and not item["has_section_heading_match"] for item in items
    )
    section_only = sum(
        item["has_section_heading_match"] and not item["has_caption_match"] for item in items
    )
    neither_source_field = sum(
        not item["has_caption_match"] and not item["has_section_heading_match"]
        for item in items
    )
    marked_caption_matches = sum(
        item["has_caption_match"] and item["has_float_selected_marker"] for item in items
    )
    unmarked_caption_matches = sum(
        item["has_caption_match"] and not item["has_float_selected_marker"] for item in items
    )
    selected = select_examples(items)
    lines = [
        "# Diagnostic Analysis of Unmapped Qasper Evidence",
        "",
        "This is a deterministic text-matching diagnostic, not manual adjudication. It does not change the exact evidence mapping or retrieval scores. A heuristic candidate is a pointer for later review, not a recovered gold label.",
        "",
        "## Scope",
        "",
        f"- Questions: {len(questions):,}",
        f"- Paragraph units searched within the matching paper: {len(chunks):,}",
        f"- Annotated evidence items: {total_evidence_items:,}",
        f"- Unmapped evidence items: {len(items):,}",
        f"- Unique normalized unmapped strings: {len(unique_texts):,}",
        f"- Evidence items with a `FLOAT SELECTED:` prefix: {marker_count:,}",
        f"- Separate figure/table captions available for comparison: {len(captions):,}",
        f"- Source section headings available for comparison: {len(sections or []):,}",
        f"- Unmapped evidence items that exactly match a caption after optional marker removal: {caption_matches:,}",
        f"- Unmapped evidence items that exactly match a same-paper section heading: {section_matches:,}",
        f"- Section-heading matches whose source section has no nonempty paragraph: {section_matches_without_body:,}",
        "",
        "## Separate figure/table caption lookup",
        "",
        "Captions are compared only within the evidence item's paper, using the same NFKC, whitespace-cleanup, and case-folding rule as exact paragraph mapping. For a leading `FLOAT SELECTED:` annotation, that marker is removed before comparing. Caption matches are diagnostic and are not added to paragraph evidence or retrieval scores.",
        "",
        "| Evidence group | Unmapped evidence items | Exact caption matches |",
        "|---|---:|---:|",
        f"| With `FLOAT SELECTED:` marker | {marker_count:,} | {marked_caption_matches:,} |",
        f"| Without marker | {len(items) - marker_count:,} | {unmarked_caption_matches:,} |",
        f"| Total | {len(items):,} | {caption_matches:,} |",
        "",
        "## Exact source-field disposition",
        "",
        "This table partitions unmapped evidence by exact matches in separately exported caption and original source section-heading fields; these fields are not paragraph retrieval hits.",
        "",
        "| Disposition | Evidence items |",
        "|---|---:|",
        f"| Caption only | {caption_only:,} |",
        f"| Section heading only | {section_only:,} |",
        f"| Both fields | {caption_and_section:,} |",
        f"| Neither field | {neither_source_field:,} |",
        "",
        "A matching heading may have an empty source body. Such a match identifies annotation provenance only and does not recover a paragraph.",
        "",
        "## Heuristic diagnostic categories",
        "",
        "The source mapping uses NFKC normalization, whitespace cleanup, and case-folding. The additional compact comparison below removes whitespace, punctuation, and symbols to locate possible candidates. It is deliberately more permissive and is not used as a mapping rule.",
        "",
        "| Diagnostic category | Evidence items | Distinct evidence strings | With `FLOAT SELECTED:` |",
        "|---|---:|---:|---:|",
    ]
    unique_by_category: dict[str, set[str]] = defaultdict(set)
    for item in items:
        unique_by_category[item["diagnostic_category"]].add(item["evidence_key"])
    for category in sorted(
        set(category_counts) | set(unique_by_category) | set(DIAGNOSTIC_CATEGORIES)
    ):
        lines.append(
            f"| `{category}` | {category_counts[category]:,} | "
            f"{len(unique_by_category[category]):,} | {marked_by_category[category]:,} |"
        )
    lines.extend(
        [
            "",
            "`marker_stripped_full_text_candidate` means removing a leading `FLOAT SELECTED:` marker, then removing non-alphanumeric characters, makes the evidence identical to a paragraph. `punctuation_or_symbol_only_full_text_candidate` means non-alphanumeric character removal alone makes them identical. `unique_long_substring_candidate` and `multiple_long_substring_candidates` require at least 60 alphanumeric characters in both the evidence and candidate paragraph, then allow containment in either direction. `no_compact_candidate_found` means these checks found no same-paper paragraph candidate. These categories may reflect table/figure evidence, incomplete excerpts, conversion artifacts, or alignment differences; they do not establish the cause.",
            "",
            (
                f"Under these fixed checks, {category_counts['marker_stripped_full_text_candidate'] + category_counts['punctuation_or_symbol_only_full_text_candidate']} items matched a whole paragraph, "
                f"{category_counts['unique_long_substring_candidate'] + category_counts['multiple_long_substring_candidates']} had at least one long substring candidate, and {category_counts['no_compact_candidate_found']} had no candidate. "
                "This is a search summary, not evidence that the remaining source text is absent."
            ),
            "",
            "## Paragraph-candidate examples",
            "",
            "Up to three examples per category are selected by sorting SHA-256 hashes of seed, question ID, annotation index, and normalized evidence text. Candidate IDs are shown only where the heuristic found them.",
        ]
    )
    for category, examples in selected.items():
        lines.extend(["", f"### `{category}`", ""])
        for item in examples:
            candidate_text = (
                ", ".join(f"`{value}`" for value in item["candidate_paragraph_ids"])
                if item["candidate_paragraph_ids"]
                else "none found"
            )
            marker = "yes" if item["has_float_selected_marker"] else "no"
            lines.extend(
                [
                    f"- Paper `{item['paper_id']}`, question `{item['question_id']}`, annotation {item['annotation_index']}; `FLOAT SELECTED:` marker: {marker}; candidate paragraph(s): {candidate_text}.",
                    "",
                    f"  > {_excerpt(item['evidence_text'])}",
                ]
            )
    caption_examples = [item for item in items if item["has_caption_match"]]
    caption_examples.sort(
        key=lambda item: hashlib.sha256(
            f"{SAMPLE_SEED}:caption:{item['question_id']}:{item['annotation_index']}:{item['evidence_key']}".encode()
        ).hexdigest()
    )
    if caption_examples:
        lines.extend(["", "## Caption-match examples", ""])
        captions_by_id = {caption["caption_id"]: caption for caption in captions}
        for item in caption_examples[:3]:
            caption_id = item["caption_match_ids"][0]
            caption_text = captions_by_id[caption_id]["text"]
            lines.extend(
                [
                    f"- Paper `{item['paper_id']}`, question `{item['question_id']}`; caption `{caption_id}`.",
                    "",
                    f"  Evidence: {_excerpt(item['evidence_text'])}",
                    "",
                    f"  Caption: {_excerpt(caption_text)}",
                ]
            )
    lines.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "No unmapped item is promoted into the official mapped-evidence set by this analysis. A section-heading match describes source metadata, not a paragraph retrieved by the evaluated systems. Matches to sections with no nonempty paragraph indicate a source heading with no body available in the converted record; they do not supply retrievable evidence. Any proposed paragraph mapping still requires a separate review and a defensible mapping rule. The full item-level diagnostics and candidate IDs are saved locally at `results/experiments/unmapped_evidence_analysis/diagnostics.jsonl` with input hashes in `manifest.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose unmapped Qasper evidence with conservative same-paper text candidates."
    )
    parser.add_argument(
        "--question-file", type=Path, default=Path("data/processed/full_validation/evaluation_questions.jsonl")
    )
    parser.add_argument(
        "--paragraph-file", type=Path, default=Path("data/processed/full_validation/evaluation_chunks.jsonl")
    )
    parser.add_argument(
        "--caption-file", type=Path, default=Path("data/processed/full_validation/evaluation_captions.jsonl")
    )
    parser.add_argument(
        "--section-file", type=Path, default=Path("data/processed/full_validation/evaluation_sections.jsonl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/unmapped_evidence_analysis.md")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/experiments/unmapped_evidence_analysis")
    )
    args = parser.parse_args()
    questions = read_jsonl(args.question_file)
    chunks = read_jsonl(args.paragraph_file)
    captions = read_jsonl(args.caption_file)
    sections = read_jsonl(args.section_file) if args.section_file.exists() else None
    items = collect_unmapped_items(questions, chunks, captions, sections)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "diagnostics.jsonl"
    with diagnostics_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(questions, chunks, captions, items, sections), encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "question_file": str(args.question_file),
        "question_file_sha256": sha256_file(args.question_file),
        "paragraph_file": str(args.paragraph_file),
        "paragraph_file_sha256": sha256_file(args.paragraph_file),
        "caption_file": str(args.caption_file),
        "caption_file_sha256": sha256_file(args.caption_file),
        "section_file": str(args.section_file) if sections is not None else None,
        "section_file_sha256": sha256_file(args.section_file) if sections is not None else None,
        "report_file": str(report),
        "report_sha256": sha256_file(report),
        "diagnostics_file": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "question_count": len(questions),
        "paragraph_count": len(chunks),
        "caption_count": len(captions),
        "section_heading_count": len(sections) if sections is not None else None,
        "unmapped_evidence_item_count": len(items),
        "exact_caption_match_count": sum(item["has_caption_match"] for item in items),
        "exact_caption_match_with_float_marker_count": sum(
            item["has_caption_match"] and item["has_float_selected_marker"] for item in items
        ),
        "exact_section_heading_match_count": sum(
            item["has_section_heading_match"] for item in items
        ),
        "exact_caption_and_section_match_count": sum(
            item["has_caption_match"] and item["has_section_heading_match"] for item in items
        ),
        "exact_section_heading_match_without_body_count": sum(
            bool(item["section_heading_matches_without_body"]) for item in items
        ),
        "distinct_unmapped_text_count": len({item["evidence_key"] for item in items}),
        "official_mapping_changed": False,
        "diagnostics_are_human_adjudications": False,
        "sample_seed": SAMPLE_SEED,
        "substring_min_alphanumeric_length": MIN_SUBSTRING_LENGTH,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report), **manifest}, indent=2))


if __name__ == "__main__":
    main()
