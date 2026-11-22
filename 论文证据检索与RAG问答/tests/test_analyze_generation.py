import json
import sys

import pytest

from scripts.analyze_generation import main, select_paired_retrieval_rows


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_analyze_generation_computes_paired_paper_cluster_differences(
    tmp_path, monkeypatch, capsys
):
    questions = [
        {
            "question_id": "q1",
            "paper_id": "paper-a",
            "annotations": [{"unanswerable": False, "reference_answers": ["yes"], "mapped_evidence_ids": ["p1"]}],
        },
        {
            "question_id": "q2",
            "paper_id": "paper-b",
            "annotations": [{"unanswerable": False, "reference_answers": ["no"], "mapped_evidence_ids": ["p2"]}],
        },
    ]
    left = [
        {"question_id": "q1", "answer": "yes", "abstained": False, "cited_paragraph_ids": ["p1"], "model": "offline-a"},
        {"question_id": "q2", "answer": "yes", "abstained": False, "cited_paragraph_ids": ["p2"], "model": "offline-a"},
    ]
    right = [
        {"question_id": "q1", "answer": "yes", "abstained": False, "cited_paragraph_ids": ["p1"], "model": "offline-b"},
        {"question_id": "q2", "answer": "no", "abstained": False, "cited_paragraph_ids": ["p2"], "model": "offline-b"},
    ]
    question_path = tmp_path / "questions.jsonl"
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    output_path = tmp_path / "comparison.json"
    write_jsonl(question_path, questions)
    write_jsonl(left_path, left)
    write_jsonl(right_path, right)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_generation.py",
            "--question-file",
            str(question_path),
            "--method",
            f"left={left_path}",
            "--method",
            f"right={right_path}",
            "--bootstrap-replicates",
            "100",
            "--output",
            str(output_path),
        ],
    )

    main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    comparison = result["paired_comparisons"]["left_vs_right"]["metrics"]["token_f1"]
    assert result["protocol"]["question_count"] == 2
    assert comparison["n_questions"] == 2
    assert comparison["n_clusters"] == 2
    assert comparison["mean_difference"] == pytest.approx(0.5)
    assert comparison["ci_low"] <= comparison["mean_difference"] <= comparison["ci_high"]
    assert '"comparison_count": 1' in capsys.readouterr().out


def test_analyze_generation_rejects_unpaired_question_sets(tmp_path, monkeypatch):
    question_path = tmp_path / "questions.jsonl"
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    question_path.write_text(
        json.dumps({"question_id": "q1", "paper_id": "paper", "annotations": []}) + "\n",
        encoding="utf-8",
    )
    base = {"answer": "", "abstained": True, "cited_paragraph_ids": []}
    write_jsonl(left_path, [{**base, "question_id": "q1"}])
    write_jsonl(right_path, [{**base, "question_id": "q2"}])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_generation.py",
            "--question-file",
            str(question_path),
            "--method",
            f"left={left_path}",
            "--method",
            f"right={right_path}",
            "--output",
            str(tmp_path / "comparison.json"),
        ],
    )

    with pytest.raises(ValueError, match="same question IDs"):
        main()


def test_analyze_generation_accepts_full_retrieval_file_for_generated_subset(
    tmp_path, monkeypatch
):
    question_path = tmp_path / "questions.jsonl"
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    output_path = tmp_path / "comparison.json"
    questions = [
        {
            "question_id": question_id,
            "paper_id": "paper",
            "annotations": [
                {
                    "unanswerable": False,
                    "reference_answers": ["yes"],
                    "mapped_evidence_ids": ["p1"],
                }
            ],
        }
        for question_id in ("selected", "not-generated")
    ]
    prediction = {
        "question_id": "selected",
        "answer": "yes",
        "abstained": False,
        "cited_paragraph_ids": ["p1"],
    }
    write_jsonl(question_path, questions)
    write_jsonl(left_path, [prediction])
    write_jsonl(right_path, [prediction])
    write_jsonl(
        retrieval_path,
        [
            {"question_id": "selected", "retrieved": [{"paragraph_id": "p1"}]},
            {"question_id": "not-generated", "retrieved": []},
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_generation.py",
            "--question-file", str(question_path),
            "--method", f"left={left_path}",
            "--method", f"right={right_path}",
            "--retrieval", f"left={retrieval_path}",
            "--retrieval", f"right={retrieval_path}",
            "--bootstrap-replicates", "10",
            "--output", str(output_path),
        ],
    )

    main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["protocol"]["question_count"] == 1
    for method in ("left", "right"):
        assert result["method_metrics"][method]["citation_metrics"]["groundedness_micro"] == 1.0
        assert [row["question_id"] for row in result["per_question_metrics"][method]] == ["selected"]


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        (
            [{"question_id": "selected"}, {"question_id": "extra"}, {"question_id": "extra"}],
            "duplicate question IDs",
        ),
        ([{"question_id": "selected"}, {"question_id": "unknown"}], "unknown question IDs"),
        ([{"question_id": "extra"}], "lacks paired questions"),
        ([{"question_id": "selected"}, {}], "no valid question_id"),
    ],
)
def test_retrieval_validation_happens_before_subset_selection(rows, error):
    with pytest.raises(ValueError, match=error):
        select_paired_retrieval_rows(
            rows, "dense", {"selected", "extra"}, {"selected"}
        )
