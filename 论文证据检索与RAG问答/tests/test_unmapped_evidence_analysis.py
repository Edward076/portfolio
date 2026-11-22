from scripts.analyze_unmapped_evidence import classify_unmapped_evidence, collect_unmapped_items


def test_compact_match_is_reported_as_candidate_not_mapped_evidence():
    paragraphs = [
        {"paragraph_id": "p1", "text": "This result is precise: 42%."},
        {"paragraph_id": "p2", "text": "A different paragraph."},
    ]

    result = classify_unmapped_evidence("This result is precise 42%", paragraphs)

    assert result["diagnostic_category"] == "punctuation_or_symbol_only_full_text_candidate"
    assert result["candidate_paragraph_ids"] == ["p1"]


def test_long_unique_substring_and_extraction_marker_are_reported():
    evidence = (
        "FLOAT SELECTED: The experiment evaluates precision and recall using the test labels "
        "across diverse experimental conditions"
    )
    paragraphs = [
        {
            "paragraph_id": "p3",
            "text": "Results show that the experiment evaluates precision and recall using the test labels across diverse experimental conditions and several settings.",
        },
        {"paragraph_id": "p4", "text": "Unrelated content."},
    ]

    result = classify_unmapped_evidence(evidence, paragraphs)

    assert result["diagnostic_category"] == "unique_long_substring_candidate"
    assert result["candidate_paragraph_ids"] == ["p3"]
    assert result["has_float_selected_marker"] is True


def test_unmapped_collection_preserves_question_and_annotation_provenance():
    questions = [
        {
            "paper_id": "paper-1",
            "question_id": "q1",
            "annotations": [
                {
                    "annotation_index": 2,
                    "unmapped_evidence_texts": ["Evidence that is not in this paragraph."],
                }
            ],
        }
    ]
    paragraphs = [
        {"paper_id": "paper-1", "paragraph_id": "p1", "text": "A paragraph."},
        {"paper_id": "paper-2", "paragraph_id": "p2", "text": "Other paper text."},
    ]

    rows = collect_unmapped_items(questions, paragraphs)

    assert len(rows) == 1
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["annotation_index"] == 2
    assert rows[0]["diagnostic_category"] == "no_compact_candidate_found"
    assert rows[0]["candidate_paragraph_ids"] == []


def test_long_evidence_does_not_match_short_generic_paragraph_fragment():
    evidence = "FLOAT SELECTED: A lengthy evidence string with many different characters and details"
    paragraphs = [
        {"paragraph_id": "p1", "text": "and"},
        {"paragraph_id": "p2", "text": "many"},
    ]

    result = classify_unmapped_evidence(evidence, paragraphs)

    assert result["diagnostic_category"] == "no_compact_candidate_found"
    assert result["candidate_paragraph_ids"] == []


def test_caption_match_is_separate_from_paragraph_retrieval_candidates():
    questions = [
        {
            "paper_id": "paper-1",
            "question_id": "q1",
            "annotations": [
                {
                    "annotation_index": 0,
                    "unmapped_evidence_texts": ["FLOAT SELECTED: Table 1: Main result."],
                }
            ],
        }
    ]
    paragraphs = [{"paper_id": "paper-1", "paragraph_id": "p1", "text": "Body text."}]
    captions = [
        {
            "paper_id": "paper-1",
            "caption_id": "paper-1:ft000",
            "text": "Table 1: Main result.",
        }
    ]

    rows = collect_unmapped_items(questions, paragraphs, captions)

    assert rows[0]["has_caption_match"] is True
    assert rows[0]["caption_match_ids"] == ["paper-1:ft000"]
    assert rows[0]["candidate_paragraph_ids"] == []


def test_section_heading_match_is_recorded_separately_from_paragraph_evidence():
    questions = [
        {
            "paper_id": "paper-1",
            "question_id": "q1",
            "annotations": [
                {
                    "annotation_index": 0,
                    "unmapped_evidence_texts": ["Experimental Setup"],
                }
            ],
        }
    ]
    paragraphs = [
        {
            "paper_id": "paper-1",
            "paragraph_id": "paper-1:p003-000",
            "section_name": "Experimental Setup",
            "text": "Body text under the heading.",
        }
    ]

    sections = [
        {
            "section_id": "paper-1:s003",
            "paper_id": "paper-1",
            "text": "Experimental Setup",
            "nonempty_paragraph_count": 0,
        }
    ]

    rows = collect_unmapped_items(questions, paragraphs, sections=sections)

    assert rows[0]["has_section_heading_match"] is True
    assert rows[0]["section_heading_matches"] == ["Experimental Setup"]
    assert rows[0]["section_heading_match_ids"] == ["paper-1:s003"]
    assert rows[0]["section_heading_matches_without_body"] == ["paper-1:s003"]
    assert rows[0]["has_caption_match"] is False
    assert rows[0]["candidate_paragraph_ids"] == []
