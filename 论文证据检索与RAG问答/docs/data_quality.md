# Qasper Data Quality

Prepared with sampling seed 42 from Hugging Face source revision `fdc9d8214fbab5dd782958601db4d678e6934a54` and Parquet conversion revision `06806e4608976fc2fac0a090ac425d5b2b29caf4`.

| Split | Papers | Questions | Answerable questions | Fully unanswerable | Mixed answerability | Evidence-mapped answerable questions | Evidence items mapped / total | Excluded for alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development (`train`) | 15 | 43 | 38 | 5 | 0 | 36 / 38 (94.7%) | 51 / 57 (89.5%) | 0 |
| Evaluation (`validation` subset) | 50 | 181 | 174 | 7 | 10 | 163 / 174 (93.7%) | 454 / 539 (84.2%) | 0 |

Answer types are not mutually exclusive because a question may have multiple annotator variants: evaluation has 112 questions with extractive spans, 79 with free-form answers, 19 with yes/no labels, and 17 with at least one unanswerable annotation. Of the 181 evaluation questions, 7 are fully unanswerable and 10 mix answerable and unanswerable annotations. The remaining questions have answerable annotations only.

The evaluation subset contains 2,541 nonempty paragraph chunks. Figure/table caption counts are preserved per question; captions are not mixed into paragraph evidence. The 85 unmapped evidence items are retained in `evaluation_questions.jsonl`; they are not treated as retrieval failures. No question records were removed for question/answer alignment mismatch.

## Full validation split

The primary retrieval evaluation uses the complete validation split, prepared separately under `data/processed/full_validation/`:

| Papers | Questions | Answerable | Fully unanswerable | Mixed answerability | Mapped-evidence questions | Evidence items mapped / total | Excluded for alignment |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 281 | 1,005 | 945 | 60 | 75 | 888 / 945 (94.0%) | 2,426 / 2,808 (86.4%) | 0 |

There are 382 unmapped evidence items. The full validation set contains 13,266 nonempty paragraphs, which produce 13,315 token-aware passages for the primary retrieval runs. The deterministic 50-paper pilot results remain available in the initial `data/processed/` output and are not mixed into the full-validation report.

Exact paper IDs, question IDs, the schema, sample fingerprints, and the full exclusion list are recorded in `data/processed/dataset_manifest.json`. Full-validation figure/table captions and source section headings are stored separately in `data/processed/full_validation/evaluation_captions.jsonl` and `evaluation_sections.jsonl`; the section export retains headings even when their converted source body is empty. Both are audit metadata and stay outside retrieval paragraphs. File counts and SHA-256 hashes are recorded in the manifest. Machine-generated data files are ignored by Git and can be recreated with `scripts/prepare_dataset.py`.

An exploratory same-paper diagnostic of the 382 unmapped evidence strings is documented in [`unmapped_evidence_analysis.md`](unmapped_evidence_analysis.md). All 253 strings prefixed `FLOAT SELECTED:` exactly match a figure/table caption after removing that marker. The other 129 exactly match section-heading metadata; 2 of those headings have no nonempty paragraph in the converted source. None of the 382 has a paragraph candidate under the fixed diagnostic checks. Caption and heading matches explain source-field provenance, but remain outside paragraph mapping and do not alter the scored question population or retrieval results. In particular, an empty heading body is not a recoverable passage.
