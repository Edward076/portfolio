# Diagnostic Analysis of Unmapped Qasper Evidence

This is a deterministic text-matching diagnostic, not manual adjudication. It does not change the exact evidence mapping or retrieval scores. A heuristic candidate is a pointer for later review, not a recovered gold label.

## Scope

- Questions: 1,005
- Paragraph units searched within the matching paper: 13,266
- Annotated evidence items: 2,808
- Unmapped evidence items: 382
- Unique normalized unmapped strings: 259
- Evidence items with a `FLOAT SELECTED:` prefix: 253
- Separate figure/table captions available for comparison: 1,929
- Source section headings available for comparison: 3,796
- Unmapped evidence items that exactly match a caption after optional marker removal: 253
- Unmapped evidence items that exactly match a same-paper section heading: 129
- Section-heading matches whose source section has no nonempty paragraph: 2

## Separate figure/table caption lookup

Captions are compared only within the evidence item's paper, using the same NFKC, whitespace-cleanup, and case-folding rule as exact paragraph mapping. For a leading `FLOAT SELECTED:` annotation, that marker is removed before comparing. Caption matches are diagnostic and are not added to paragraph evidence or retrieval scores.

| Evidence group | Unmapped evidence items | Exact caption matches |
|---|---:|---:|
| With `FLOAT SELECTED:` marker | 253 | 253 |
| Without marker | 129 | 0 |
| Total | 382 | 253 |

## Exact source-field disposition

This table partitions unmapped evidence by exact matches in separately exported caption and original source section-heading fields; these fields are not paragraph retrieval hits.

| Disposition | Evidence items |
|---|---:|
| Caption only | 253 |
| Section heading only | 129 |
| Both fields | 0 |
| Neither field | 0 |

A matching heading may have an empty source body. Such a match identifies annotation provenance only and does not recover a paragraph.

## Heuristic diagnostic categories

The source mapping uses NFKC normalization, whitespace cleanup, and case-folding. The additional compact comparison below removes whitespace, punctuation, and symbols to locate possible candidates. It is deliberately more permissive and is not used as a mapping rule.

| Diagnostic category | Evidence items | Distinct evidence strings | With `FLOAT SELECTED:` |
|---|---:|---:|---:|
| `marker_stripped_full_text_candidate` | 0 | 0 | 0 |
| `multiple_long_substring_candidates` | 0 | 0 | 0 |
| `no_compact_candidate_found` | 382 | 259 | 253 |
| `punctuation_or_symbol_only_full_text_candidate` | 0 | 0 | 0 |
| `unique_long_substring_candidate` | 0 | 0 | 0 |

`marker_stripped_full_text_candidate` means removing a leading `FLOAT SELECTED:` marker, then removing non-alphanumeric characters, makes the evidence identical to a paragraph. `punctuation_or_symbol_only_full_text_candidate` means non-alphanumeric character removal alone makes them identical. `unique_long_substring_candidate` and `multiple_long_substring_candidates` require at least 60 alphanumeric characters in both the evidence and candidate paragraph, then allow containment in either direction. `no_compact_candidate_found` means these checks found no same-paper paragraph candidate. These categories may reflect table/figure evidence, incomplete excerpts, conversion artifacts, or alignment differences; they do not establish the cause.

Under these fixed checks, 0 items matched a whole paragraph, 0 had at least one long substring candidate, and 382 had no candidate. This is a search summary, not evidence that the remaining source text is absent.

## Paragraph-candidate examples

Up to three examples per category are selected by sorting SHA-256 hashes of seed, question ID, annotation index, and normalized evidence text. Candidate IDs are shown only where the heuristic found them.

### `no_compact_candidate_found`

- Paper `1703.10344`, question `77c34f1033702278f7f044806c1eba0c6ecb8b04`, annotation 0; `FLOAT SELECTED:` marker: yes; candidate paragraph(s): none found.

  > FLOAT SELECTED: Figure 1: Comparing how cyclones are reported in Wikipedia entity pages.
- Paper `1910.09982`, question `6bfba3ddca5101ed15256fca75fcdc95a53cece7`, annotation 1; `FLOAT SELECTED:` marker: no; candidate paragraph(s): none found.

  > Propaganda Techniques ::: 8. Causal oversimplification.
- Paper `1909.08089`, question `de5b6c25e35b3a6c5e40e350fc5e52c160b33490`, annotation 1; `FLOAT SELECTED:` marker: yes; candidate paragraph(s): none found.

  > FLOAT SELECTED: Table 4.2: Results on the Pubmed dataset. For models with an ∗, we report results from [8]. See caption of Table 4.1 above for details on compared models. Results that are not significantly distinguished from the best systems are bold.

## Caption-match examples

- Paper `1909.00015`, question `f8c1b17d265a61502347c9a937269b38fc3fcab1`; caption `1909.00015:ft001`.

  Evidence: FLOAT SELECTED: Table 1: Machine translation tokenized BLEU test results on IWSLT 2017 DE EN, KFTT JA EN, WMT 2016 RO EN and WMT 2014 EN DE, respectively.

  Caption: Table 1: Machine translation tokenized BLEU test results on IWSLT 2017 DE EN, KFTT JA EN, WMT 2016 RO EN and WMT 2014 EN DE, respectively.
- Paper `2002.11910`, question `9da1e124d28b488b0d94998d32aa2fa8a5ebec51`; caption `2002.11910:ft001`.

  Evidence: FLOAT SELECTED: Table 1: The results of two previous models, and results of this study, in which we apply a boundary assembling method. Precision, recall, and F1 scores are shown for both named entity and nominal mention. For both tasks and their overall performance, we outperform the other two models.

  Caption: Table 1: The results of two previous models, and results of this study, in which we apply a boundary assembling method. Precision, recall, and F1 scores are shown for both named entity and nominal mention. For both tasks and their overall performance, we outperform the other two models.
- Paper `1809.02286`, question `0ad4359e3e7e5e5f261c2668fe84c12bc762b3b8`; caption `1809.02286:ft002`.

  Evidence: FLOAT SELECTED: Table 1: The comparison of various models on different sentence classification tasks. We report the test accuracy of each model in percentage. Our SATA Tree-LSTM shows superior or competitive performance on all tasks, compared to previous treestructured models as well as other sophisticated models. ?: Latent tree-structured models. †: Models…

  Caption: Table 1: The comparison of various models on different sentence classification tasks. We report the test accuracy of each model in percentage. Our SATA Tree-LSTM shows superior or competitive performance on all tasks, compared to previous treestructured models as well as other sophisticated models. ?: Latent tree-structured models. †: Models which are pre-tr…

## Interpretation limit

No unmapped item is promoted into the official mapped-evidence set by this analysis. A section-heading match describes source metadata, not a paragraph retrieved by the evaluated systems. Matches to sections with no nonempty paragraph indicate a source heading with no body available in the converted record; they do not supply retrievable evidence. Any proposed paragraph mapping still requires a separate review and a defensible mapping rule. The full item-level diagnostics and candidate IDs are saved locally at `results/experiments/unmapped_evidence_analysis/diagnostics.jsonl` with input hashes in `manifest.json`.
