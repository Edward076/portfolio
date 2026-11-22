from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(char if char in "\n\t" or char.isprintable() else " " for char in text)
    return re.sub(r"\s+", " ", text).strip()


def evidence_key(text: str) -> str:
    return clean_text(text).casefold()


def flatten_figure_table_captions(paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract captions as a separate record type; never mix them into paragraphs."""
    paper_id = str(paper["id"])
    title = clean_text(paper.get("title"))
    caption_values = (paper.get("figures_and_tables") or {}).get("caption") or []
    rows = []
    for caption_index, raw_caption in enumerate(caption_values):
        text = clean_text(raw_caption)
        if not text:
            continue
        rows.append(
            {
                "caption_id": f"{paper_id}:ft{caption_index:03d}",
                "paper_id": paper_id,
                "paper_title": title,
                "caption_index": caption_index,
                "text": text,
            }
        )
    return rows


def flatten_section_headings(paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve source section labels, including labels whose body is empty."""
    paper_id = str(paper["id"])
    title = clean_text(paper.get("title"))
    full_text = paper.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    section_paragraphs = full_text.get("paragraphs") or []
    rows = []
    for section_index, raw_name in enumerate(section_names):
        text = clean_text(raw_name)
        paragraphs = section_paragraphs[section_index] if section_index < len(section_paragraphs) else []
        nonempty_paragraph_count = sum(bool(clean_text(value)) for value in (paragraphs or []))
        rows.append(
            {
                "section_id": f"{paper_id}:s{section_index:03d}",
                "paper_id": paper_id,
                "paper_title": title,
                "section_index": section_index,
                "text": text,
                "nonempty_paragraph_count": nonempty_paragraph_count,
            }
        )
    return rows


def flatten_paper(
    paper: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return chunks, aligned questions, and excluded-question records."""
    paper_id = str(paper["id"])
    title = clean_text(paper.get("title"))
    full_text = paper.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    section_paragraphs = full_text.get("paragraphs") or []
    chunks: list[dict[str, Any]] = []
    paragraph_lookup: dict[str, list[int]] = {}
    for section_index, paragraphs in enumerate(section_paragraphs):
        section = (
            clean_text(section_names[section_index]) if section_index < len(section_names) else ""
        )
        for local_index, raw_text in enumerate(paragraphs or []):
            text = clean_text(raw_text)
            if not text:
                continue
            paragraph_id = f"{paper_id}:p{section_index:03d}-{local_index:03d}"
            chunk = {
                "chunk_id": paragraph_id,
                "paper_id": paper_id,
                "paper_title": title,
                "section_name": section,
                "paragraph_id": paragraph_id,
                "paragraph_index": len(chunks),
                "text": text,
            }
            chunks.append(chunk)
            paragraph_lookup.setdefault(evidence_key(text), []).append(len(chunks) - 1)

    qas = paper.get("qas") or {}
    questions = qas.get("question") or []
    question_ids = qas.get("question_id") or []
    answer_records = qas.get("answers") or []
    excluded: list[dict[str, Any]] = []
    aligned: list[dict[str, Any]] = []
    if len(questions) != len(question_ids) or len(questions) != len(answer_records):
        excluded.append(
            {
                "paper_id": paper_id,
                "reason": "question_id_or_answer_record_count_mismatch",
                "questions": len(questions),
                "question_ids": len(question_ids),
                "answer_records": len(answer_records),
            }
        )

    for index, question in enumerate(questions):
        if index >= len(question_ids) or index >= len(answer_records):
            excluded.append(
                {"paper_id": paper_id, "question_index": index, "reason": "unaligned_question"}
            )
            continue
        record = answer_records[index] or {}
        variants = record.get("answer") or []
        if not clean_text(question) or not variants:
            excluded.append(
                {
                    "paper_id": paper_id,
                    "question_id": question_ids[index],
                    "reason": "empty_question_or_annotations",
                }
            )
            continue
        annotations: list[dict[str, Any]] = []
        for variant_index, variant in enumerate(variants):
            variant = variant or {}
            evidences = [
                clean_text(value) for value in (variant.get("evidence") or []) if clean_text(value)
            ]
            mapped_ids: list[str] = []
            unmapped: list[str] = []
            for evidence in evidences:
                exact_matches = paragraph_lookup.get(evidence_key(evidence), [])
                if exact_matches:
                    mapped_ids.extend(chunks[pos]["paragraph_id"] for pos in exact_matches)
                else:
                    unmapped.append(evidence)
            yes_no = variant.get("yes_no")
            extractive = [
                clean_text(v) for v in (variant.get("extractive_spans") or []) if clean_text(v)
            ]
            free_form = clean_text(variant.get("free_form_answer"))
            if variant.get("unanswerable"):
                answer_type = "unanswerable"
            elif yes_no is not None:
                answer_type = "yes_no"
            elif extractive:
                answer_type = "extractive"
            else:
                answer_type = "free_form"
            annotations.append(
                {
                    "annotation_index": variant_index,
                    "answer_type": answer_type,
                    "unanswerable": bool(variant.get("unanswerable")),
                    "yes_no": yes_no,
                    "extractive_spans": extractive,
                    "free_form_answer": free_form,
                    "reference_answers": sorted(
                        set(
                            extractive
                            + ([free_form] if free_form else [])
                            + ([str(yes_no)] if yes_no is not None else [])
                        )
                    ),
                    "evidence_texts": evidences,
                    "mapped_evidence_ids": sorted(set(mapped_ids)),
                    "unmapped_evidence_texts": unmapped,
                }
            )
        aligned.append(
            {
                "paper_id": paper_id,
                "paper_title": title,
                "question_id": str(question_ids[index]),
                "question": clean_text(question),
                "annotations": annotations,
                "figure_table_count": len(
                    (paper.get("figures_and_tables") or {}).get("caption", [])
                ),
            }
        )
    return chunks, aligned, excluded


def sample_papers(dataset: Iterable[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    papers = list(dataset)
    random.Random(seed).shuffle(papers)
    return papers[:count]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def summarize_quality(
    papers: list[dict[str, Any]], questions: list[dict[str, Any]], excluded: list[dict[str, Any]]
) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    answerable = 0
    mapped_questions = 0
    total_annotated_questions = 0
    evidence_total = 0
    evidence_mapped = 0
    evidence_unmapped = 0
    fully_unanswerable_questions = 0
    mixed_answerability_questions = 0
    for question in questions:
        annotations = question["annotations"]
        types = {a["answer_type"] for a in annotations}
        type_counts.update(types)
        answerability = {a["unanswerable"] for a in annotations}
        if answerability == {True}:
            fully_unanswerable_questions += 1
        elif answerability == {True, False}:
            mixed_answerability_questions += 1
        if any(not a["unanswerable"] for a in annotations):
            answerable += 1
            total_annotated_questions += 1
            if any(a["mapped_evidence_ids"] for a in annotations if not a["unanswerable"]):
                mapped_questions += 1
        for annotation in annotations:
            if annotation["unanswerable"]:
                continue
            evidence_total += len(annotation["evidence_texts"])
            evidence_mapped += len(annotation["evidence_texts"]) - len(
                annotation["unmapped_evidence_texts"]
            )
            evidence_unmapped += len(annotation["unmapped_evidence_texts"])
    return {
        "paper_count": len(papers),
        "question_count": len(questions),
        "answerable_question_count": answerable,
        "fully_unanswerable_question_count": fully_unanswerable_questions,
        "fully_unanswerable_question_rate": fully_unanswerable_questions / len(questions)
        if questions
        else None,
        "mixed_answerability_question_count": mixed_answerability_questions,
        "answer_type_question_counts_nonexclusive": dict(sorted(type_counts.items())),
        "answerable_question_evidence_mapped_count": mapped_questions,
        "answerable_question_evidence_mapping_rate": mapped_questions / total_annotated_questions
        if total_annotated_questions
        else None,
        "annotated_evidence_item_count": evidence_total,
        "mapped_evidence_item_count": evidence_mapped,
        "unmapped_evidence_item_count": evidence_unmapped,
        "evidence_item_mapping_rate": evidence_mapped / evidence_total if evidence_total else None,
        "excluded_record_count": len(excluded),
        "excluded_reasons": dict(Counter(row["reason"] for row in excluded)),
        "paper_ids": sorted(str(paper["id"]) for paper in papers),
        "question_ids": sorted(question["question_id"] for question in questions),
        "sample_fingerprint_sha256": sha256_json(
            {
                "paper_ids": sorted(str(paper["id"]) for paper in papers),
                "question_ids": sorted(question["question_id"] for question in questions),
            }
        ),
    }
