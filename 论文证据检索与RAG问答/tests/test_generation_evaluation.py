from __future__ import annotations

import pytest

from scientific_rag.generation_evaluation import evaluate_generation, normalize_answer, token_f1
from scientific_rag.generation_prompt import (
    build_closed_book_messages,
    build_generation_messages,
    parse_generation_response,
)


def test_normalize_answer_and_token_f1() -> None:
    assert normalize_answer("The quick, brown fox!") == "quick brown fox"
    assert normalize_answer("False") == "no"
    assert normalize_answer("Yes") == normalize_answer("True")
    assert token_f1("The result works.", "result works") == 1.0
    assert token_f1("red fox", "blue fox") == pytest.approx(0.5)


def test_evaluate_answerability_and_citations_with_mixed_annotations() -> None:
    questions = [
        {
            "question_id": "yes-no",
            "annotations": [
                {
                    "unanswerable": False,
                    "reference_answers": ["True"],
                    "mapped_evidence_ids": ["paper:p1"],
                }
            ],
        },
        {
            "question_id": "unanswerable",
            "annotations": [
                {"unanswerable": True, "reference_answers": [], "mapped_evidence_ids": []}
            ],
        },
        {
            "question_id": "mixed",
            "annotations": [
                {
                    "unanswerable": False,
                    "reference_answers": ["The result works"],
                    "mapped_evidence_ids": ["paper:p3"],
                },
                {"unanswerable": True, "reference_answers": [], "mapped_evidence_ids": []},
            ],
        },
    ]
    predictions = [
        {
            "question_id": "yes-no",
            "answer": "Yes",
            "abstained": False,
            "cited_paragraph_ids": ["paper:p1"],
        },
        {
            "question_id": "unanswerable",
            "answer": "",
            "abstained": True,
            "cited_paragraph_ids": [],
        },
        {
            "question_id": "mixed",
            "answer": "result works",
            "abstained": False,
            "cited_paragraph_ids": ["paper:invented"],
        },
    ]
    retrieval = [
        {
            "question_id": "yes-no",
            "retrieved": [{"chunk_id": "child1", "paragraph_id": "paper:p1"}],
        },
        {"question_id": "unanswerable", "retrieved": []},
        {
            "question_id": "mixed",
            "retrieved": [{"chunk_id": "child3", "paragraph_id": "paper:p3"}],
        },
    ]

    result = evaluate_generation(questions, predictions, retrieval, context_k=5)

    assert result["counts"] == {
        "question_count": 3,
        "prediction_count": 3,
        "missing_prediction_count": 0,
        "answer_scored_count": 2,
        "fully_unanswerable_count": 1,
        "mixed_answerability_count": 1,
        "answerability_scored_count": 2,
        "citation_scored_count": 2,
    }
    assert result["answer_metrics"]["exact_match_macro"] == 1.0
    assert result["answer_metrics"]["token_f1_macro"] == 1.0
    assert result["answerability"]["accuracy"] == 1.0
    assert result["citation_metrics"]["precision_macro"] == 0.5
    assert result["citation_metrics"]["recall_macro"] == 0.5
    assert result["citation_metrics"]["groundedness_micro"] == 0.5
    assert result["citation_metrics"]["groundedness_question_count"] == 3


@pytest.mark.parametrize("unanswerable", [True, False], ids=["unanswerable", "unmapped"])
@pytest.mark.parametrize(
    ("citation", "expected_groundedness"),
    [("paper:p1", 1.0), ("paper:invented", 0.0), ("paper:p2", 0.0)],
    ids=["shown-context", "invented-citation", "outside-top-k"],
)
def test_groundedness_does_not_require_mapped_gold_evidence(
    unanswerable: bool, citation: str, expected_groundedness: float
) -> None:
    question = {
        "question_id": "q1",
        "annotations": [
            {
                "unanswerable": unanswerable,
                "reference_answers": [] if unanswerable else ["An answer"],
                "mapped_evidence_ids": [],
            }
        ],
    }
    prediction = {
        "question_id": "q1",
        "answer": "An answer",
        "abstained": False,
        "cited_paragraph_ids": [citation],
    }
    retrieval = {
        "question_id": "q1",
        "retrieved": [
            {"paragraph_id": "paper:p1"},
            {"paragraph_id": "paper:p2"},
        ],
    }

    result = evaluate_generation([question], [prediction], [retrieval], context_k=1)

    assert result["counts"]["citation_scored_count"] == 0
    citation_metrics = result["citation_metrics"]
    assert citation_metrics["precision_macro"] is None
    assert citation_metrics["recall_macro"] is None
    assert citation_metrics["f1_macro"] is None
    assert citation_metrics["groundedness_micro"] == expected_groundedness
    assert citation_metrics["groundedness_question_count"] == 1
    assert citation_metrics["citation_count_with_retrieval_context"] == 1
    assert citation_metrics["citation_count_in_top_context"] == expected_groundedness
    per_question = result["per_question"][0]
    assert per_question["citation_precision"] is None
    assert per_question["citation_recall"] is None
    assert per_question["citation_f1"] is None
    assert per_question["groundedness_per_cited_question"] == expected_groundedness
    assert per_question["grounded_citation_count"] == expected_groundedness
    assert per_question["citation_count"] == 1


def test_groundedness_for_an_uncited_abstention_does_not_invent_a_score() -> None:
    question = {
        "question_id": "q1",
        "annotations": [{"unanswerable": True, "mapped_evidence_ids": []}],
    }
    prediction = {
        "question_id": "q1",
        "answer": "",
        "abstained": True,
        "cited_paragraph_ids": [],
    }
    retrieval = {"question_id": "q1", "retrieved": [{"paragraph_id": "paper:p1"}]}

    result = evaluate_generation([question], [prediction], [retrieval])

    assert result["citation_metrics"]["groundedness_micro"] is None
    assert result["citation_metrics"]["groundedness_question_count"] == 1
    assert result["citation_metrics"]["citation_count_with_retrieval_context"] == 0
    assert result["per_question"][0]["groundedness_per_cited_question"] is None
    assert result["per_question"][0]["grounded_citation_count"] == 0
    assert result["per_question"][0]["citation_count"] == 0


def test_prediction_schema_rejects_duplicate_or_ambiguous_records() -> None:
    question = {"question_id": "q1", "annotations": []}
    prediction = {
        "question_id": "q1",
        "answer": "answer",
        "abstained": False,
        "cited_paragraph_ids": [],
    }
    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluate_generation([question], [prediction, prediction])
    with pytest.raises(TypeError, match="abstained must be a boolean"):
        evaluate_generation([question], [{"question_id": "q1", "answer": "answer"}])


def test_generation_prompt_preserves_retrieved_chunks_and_parent_citations() -> None:
    prompt = build_generation_messages(
        "What was found?",
        [
            {"chunk_id": "c1", "paragraph_id": "p1", "section_name": "Results", "text": "First."},
            {
                "chunk_id": "c1b",
                "paragraph_id": "p1",
                "section_name": "Results",
                "text": "First, overlapped.",
            },
            {
                "chunk_id": "c2",
                "paragraph_id": "p2",
                "section_name": "Conclusion",
                "text": "Second.",
            },
        ],
        context_k=2,
    )

    assert prompt["allowed_citations"] == ["p1"]
    assert "not instructions" in prompt["messages"][1]["content"]
    assert "First, overlapped." in prompt["messages"][1]["content"]
    assert "Second." not in prompt["messages"][1]["content"]


def test_closed_book_prompt_contains_title_and_question_without_context():
    prompt = build_closed_book_messages("What did the authors find?", "A Paper")

    assert prompt["allowed_citations"] == []
    assert prompt["prompt_version"] == "qasper-closed-book-json-v1"
    assert "A Paper" in prompt["messages"][1]["content"]
    assert "What did the authors find?" in prompt["messages"][1]["content"]
    assert "Evidence passages" not in prompt["messages"][1]["content"]
    assert "without access to its article text" in prompt["messages"][0]["content"]


def test_closed_book_prompt_requires_a_title():
    with pytest.raises(ValueError, match="paper_title"):
        build_closed_book_messages("Question?", " ")


def test_generation_response_parser_enforces_citation_contract() -> None:
    parsed = parse_generation_response(
        '{"answer":"Supported answer","abstained":false,"cited_paragraph_ids":["p1","p1"]}',
        allowed_citations=["p1"],
    )
    assert parsed["cited_paragraph_ids"] == ["p1"]

    with pytest.raises(ValueError, match="not present in context"):
        parse_generation_response(
            '{"answer":"Unsupported","abstained":false,"cited_paragraph_ids":["p2"]}',
            allowed_citations=["p1"],
        )
    with pytest.raises(ValueError, match="abstentions must have"):
        parse_generation_response(
            '{"answer":"I do not know","abstained":true,"cited_paragraph_ids":[]}',
            allowed_citations=["p1"],
        )
