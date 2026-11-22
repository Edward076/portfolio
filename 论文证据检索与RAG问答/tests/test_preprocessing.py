from scientific_rag.chunking import split_token_aware
from scientific_rag.preprocessing import (
    flatten_figure_table_captions,
    flatten_paper,
    flatten_section_headings,
    sample_papers,
)


def make_paper(paper_id="p1"):
    return {
        "id": paper_id,
        "title": "A Paper",
        "abstract": "abstract",
        "full_text": {
            "section_name": ["Intro"],
            "paragraphs": [["Evidence paragraph.", "Another paragraph."]],
        },
        "qas": {
            "question": ["What is supported?"],
            "question_id": ["q1"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["supported"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["Evidence paragraph."],
                            "highlighted_evidence": [],
                        },
                        {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": None,
                            "free_form_answer": "The evidence.",
                            "evidence": ["Evidence paragraph."],
                            "highlighted_evidence": [],
                        },
                    ]
                }
            ],
        },
        "figures_and_tables": {"caption": [], "file": []},
    }


def test_qasper_nested_alignment_and_exact_evidence_mapping():
    chunks, questions, excluded = flatten_paper(make_paper())
    assert len(chunks) == 2
    assert chunks[0]["section_name"] == "Intro"
    assert questions[0]["question_id"] == "q1"
    assert len(questions[0]["annotations"]) == 2
    assert questions[0]["annotations"][0]["mapped_evidence_ids"] == ["p1:p000-000"]
    assert not excluded


def test_unmapped_evidence_is_retained():
    paper = make_paper()
    paper["qas"]["answers"][0]["answer"][0]["evidence"] = ["nonmatching annotation"]
    _, questions, _ = flatten_paper(paper)
    annotation = questions[0]["annotations"][0]
    assert annotation["mapped_evidence_ids"] == []
    assert annotation["unmapped_evidence_texts"] == ["nonmatching annotation"]


def test_figure_table_captions_are_saved_separately_from_paragraph_evidence():
    paper = make_paper()
    paper["figures_and_tables"]["caption"] = [
        "Table 1: Main results.",
        None,
        "  Figure 2:   Additional result.  ",
    ]

    captions = flatten_figure_table_captions(paper)

    assert [row["caption_id"] for row in captions] == ["p1:ft000", "p1:ft002"]
    assert [row["text"] for row in captions] == [
        "Table 1: Main results.",
        "Figure 2: Additional result.",
    ]
    chunks, _, _ = flatten_paper(paper)
    assert all(row["text"] not in {chunk["text"] for chunk in chunks} for row in captions)


def test_section_headings_preserve_empty_source_sections_separately():
    paper = make_paper()
    paper["full_text"]["section_name"] = ["Intro", "Experimental Setup"]
    paper["full_text"]["paragraphs"].append(["", "   "])

    sections = flatten_section_headings(paper)

    assert [section["section_id"] for section in sections] == ["p1:s000", "p1:s001"]
    assert [section["text"] for section in sections] == ["Intro", "Experimental Setup"]
    assert [section["nonempty_paragraph_count"] for section in sections] == [2, 0]


def test_sampling_is_reproducible():
    papers = [make_paper(f"p{i}") for i in range(20)]
    assert [p["id"] for p in sample_papers(papers, 7, 42)] == [
        p["id"] for p in sample_papers(papers, 7, 42)
    ]


def test_token_aware_chunks_retain_original_paragraph_mapping():
    class TinyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(token_ids)

    chunks = [{"chunk_id": "p1:p0", "paragraph_id": "p1:p0", "text": "a b c d e f g"}]
    split = split_token_aware(chunks, TinyTokenizer(), max_tokens=4, overlap_tokens=1)
    assert [item["chunk_id"] for item in split] == ["p1:p0:c000", "p1:p0:c001"]
    assert all(item["paragraph_id"] == "p1:p0" for item in split)
    assert split[0]["text"] == "a b c d"
    assert split[1]["text"] == "d e f g"
