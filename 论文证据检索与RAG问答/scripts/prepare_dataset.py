from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from scientific_rag.data_loader import load_qasper
from scientific_rag.preprocessing import (
    flatten_figure_table_captions,
    flatten_paper,
    flatten_section_headings,
    sample_papers,
    summarize_quality,
    write_jsonl,
)

DEFAULT_SOURCE_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
DEFAULT_PARQUET_CONVERSION_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"


def schema_manifest(dataset, source_revision: str, loading_method: str) -> dict:
    result: dict = {
        "dataset": "allenai/qasper",
        "source_revision": source_revision,
        "loading_method": loading_method,
        "splits": {},
    }
    for split_name, split in dataset.items():
        first = split[0] if len(split) else {}
        qas = first.get("qas", {})
        answer_records = qas.get("answers", []) if isinstance(qas, dict) else []
        first_record = answer_records[0] if answer_records else {}
        variants = first_record.get("answer", []) if isinstance(first_record, dict) else []
        result["splits"][split_name] = {
            "rows": len(split),
            "features": str(split.features),
            "row_keys": list(first.keys()),
            "full_text_keys": list(first.get("full_text", {}).keys()),
            "qas_keys": list(qas.keys()) if isinstance(qas, dict) else [],
            "question_count_first_paper": len(qas.get("question", []))
            if isinstance(qas, dict)
            else 0,
            "answer_record_keys": list(first_record.keys())
            if isinstance(first_record, dict)
            else [],
            "answer_variant_keys": list(variants[0].keys()) if variants else [],
        }
    return result


def prepare_split(dataset, count: int, seed: int):
    papers = sample_papers(dataset, min(count, len(dataset)), seed)
    chunks: list[dict] = []
    questions: list[dict] = []
    excluded: list[dict] = []
    for paper in papers:
        paper_chunks, paper_questions, paper_excluded = flatten_paper(paper)
        chunks.extend(paper_chunks)
        questions.extend(paper_questions)
        excluded.extend(paper_excluded)
    return papers, chunks, questions, excluded


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Inspect and prepare a deterministic Qasper subset."
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_SOURCE_REVISION,
        help="Immutable dataset source revision.",
    )
    parser.add_argument(
        "--parquet-conversion-revision",
        default=DEFAULT_PARQUET_CONVERSION_REVISION,
        help="Immutable Hugging Face Parquet conversion revision.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-papers", type=int, default=15)
    parser.add_argument("--eval-papers", type=int, default=50)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    dataset, source_revision, loading_method, conversion_revision = load_qasper(
        revision=args.revision,
        parquet_conversion_revision=args.parquet_conversion_revision,
    )
    schema = schema_manifest(dataset, source_revision, loading_method)
    schema["parquet_conversion_revision"] = conversion_revision
    if args.inspect_only:
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        return

    dev = prepare_split(dataset["train"], args.dev_papers, args.seed)
    evaluation = prepare_split(dataset["validation"], args.eval_papers, args.seed)
    dev_papers, dev_chunks, dev_questions, dev_excluded = dev
    eval_papers, eval_chunks, eval_questions, eval_excluded = evaluation
    dev_captions = [
        caption for paper in dev_papers for caption in flatten_figure_table_captions(paper)
    ]
    eval_captions = [
        caption for paper in eval_papers for caption in flatten_figure_table_captions(paper)
    ]
    dev_sections = [section for paper in dev_papers for section in flatten_section_headings(paper)]
    eval_sections = [section for paper in eval_papers for section in flatten_section_headings(paper)]
    dev_paper_ids = {str(paper["id"]) for paper in dev_papers}
    eval_paper_ids = {str(paper["id"]) for paper in eval_papers}
    if dev_paper_ids & eval_paper_ids:
        raise RuntimeError("Development and validation paper IDs overlap.")

    write_jsonl(args.output_dir / "dev_chunks.jsonl", dev_chunks)
    write_jsonl(args.output_dir / "dev_questions.jsonl", dev_questions)
    write_jsonl(args.output_dir / "evaluation_chunks.jsonl", eval_chunks)
    write_jsonl(args.output_dir / "evaluation_questions.jsonl", eval_questions)
    write_jsonl(args.output_dir / "dev_captions.jsonl", dev_captions)
    write_jsonl(args.output_dir / "evaluation_captions.jsonl", eval_captions)
    write_jsonl(args.output_dir / "dev_sections.jsonl", dev_sections)
    write_jsonl(args.output_dir / "evaluation_sections.jsonl", eval_sections)
    write_jsonl(args.output_dir / "excluded_records.jsonl", dev_excluded + eval_excluded)
    caption_files = {"development": "dev_captions.jsonl", "evaluation": "evaluation_captions.jsonl"}
    section_files = {"development": "dev_sections.jsonl", "evaluation": "evaluation_sections.jsonl"}

    report = {
        "dataset": "allenai/qasper",
        "source_revision": source_revision,
        "parquet_conversion_revision": conversion_revision,
        "loading_method": loading_method,
        "sampling_seed": args.seed,
        "runtime": {
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("datasets", "numpy", "pandas", "pyarrow")
            },
        },
        "sampling_unit": "paper",
        "official_evaluation_split": "validation",
        "evaluation_subset_label": "Qasper validation subset evaluation",
        "schema": schema,
        "development": summarize_quality(dev_papers, dev_questions, dev_excluded),
        "evaluation": summarize_quality(eval_papers, eval_questions, eval_excluded),
        "chunk_counts": {"development": len(dev_chunks), "evaluation": len(eval_chunks)},
        "caption_counts": {"development": len(dev_captions), "evaluation": len(eval_captions)},
        "section_heading_counts": {"development": len(dev_sections), "evaluation": len(eval_sections)},
        "caption_files": caption_files,
        "caption_file_sha256": {
            name: sha256_file(args.output_dir / name) for name in caption_files.values()
        },
        "section_files": section_files,
        "section_file_sha256": {
            name: sha256_file(args.output_dir / name) for name in section_files.values()
        },
        "split_paper_overlap": sorted(dev_paper_ids & eval_paper_ids),
        "known_data_handling": {
            "paragraph_ids": "paper_id:p<section-index>-<paragraph-index>",
            "evidence_matching": "exact after NFKC normalization, control cleanup, whitespace collapse, and case folding",
            "figure_table_captions": "saved separately; never treated as paragraph evidence",
            "section_headings": "saved separately, including headings with empty bodies; never treated as paragraph evidence",
            "unmapped_evidence": "retained in question annotations and reported, not scored as retrieval misses",
            "multi_annotator_rule": "all annotation variants retained; retrieval evidence sets may be evaluated against the best-matching valid annotator set",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_revision": source_revision,
                "loading_method": loading_method,
                "development": {
                    "papers": len(dev_papers),
                    "questions": len(dev_questions),
                    "chunks": len(dev_chunks),
                },
                "evaluation": {
                    "papers": len(eval_papers),
                    "questions": len(eval_questions),
                    "chunks": len(eval_chunks),
                    "captions": len(eval_captions),
                    "section_headings": len(eval_sections),
                },
                "manifest": str(args.output_dir / "dataset_manifest.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
