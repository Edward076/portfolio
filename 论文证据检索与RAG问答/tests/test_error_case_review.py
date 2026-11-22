import json

import pytest

from scripts.prepare_error_case_review import (
    build_case_records,
    main,
    make_decision_row,
    render_blinded_markdown,
)


def test_review_packet_hides_method_and_gold_labels_until_unblinding():
    question = {
        "question_id": "q1",
        "paper_id": "paper1",
        "paper_title": "Paper title",
        "question": "What is the result?",
        "annotations": [
            {
                "unanswerable": False,
                "reference_answers": ["42"],
                "mapped_evidence_ids": ["paper1:p1"],
                "unmapped_evidence_texts": [],
            }
        ],
    }
    hybrid = {
        "question_id": "q1",
        "gold_evidence_sets": [["paper1:p1"]],
        "retrieved": [
            {
                "chunk_id": f"h{i}",
                "paragraph_id": f"paper1:p{i}",
                "section_name": "Results",
                "text": f"Passage {i}",
            }
            for i in range(5)
        ],
    }
    reranked = {
        "question_id": "q1",
        "retrieved": [
            {
                "chunk_id": f"r{i}",
                "paragraph_id": f"paper1:p{i}",
                "section_name": "Results",
                "text": f"Passage {i}",
            }
            for i in range(5)
        ],
    }

    blinded, key = build_case_records(question, hybrid, reranked, "reranker_recovered", "42")

    assert "outcome_stratum" not in blinded
    assert "method_for_context_A" not in blinded
    assert "annotations" not in blinded
    assert all("score" not in passage and "rank" not in passage for passage in blinded["context_A"])
    assert all("chunk_id" in passage for passage in blinded["context_A"] + blinded["context_B"])
    assert {key["method_for_context_A"], key["method_for_context_B"]} == {
        "hybrid",
        "hybrid_reranked",
    }
    assert key["outcome_stratum"] == "reranker_recovered"
    assert key["annotations"][0]["reference_answers"] == ["42"]
    assert make_decision_row("q1")["context_A_sufficiency"] == ""
    markdown = render_blinded_markdown([blinded])
    assert "42" not in markdown
    assert "Context A" in markdown and "Context B" in markdown
    assert "outcome_stratum" not in markdown
    assert "method_for_context" not in markdown
    assert "parent paragraph" in markdown


@pytest.mark.parametrize("existing_content", [False, True])
def test_preparation_refuses_existing_directory_before_reading_inputs(
    tmp_path, monkeypatch, capsys, existing_content
):
    output = tmp_path / "review"
    output.mkdir()
    decisions = output / "review_decisions_template.csv"
    if existing_content:
        decisions.write_text("saved human judgments", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_error_case_review.py", "--root", str(tmp_path), "--output-dir", "review"],
    )
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert "Use a new --output-dir" in capsys.readouterr().err
    if existing_content:
        assert decisions.read_text(encoding="utf-8") == "saved human judgments"
    else:
        assert list(output.iterdir()) == []


def test_fresh_packet_can_be_created_but_seed_change_cannot_erase_review(
    tmp_path, monkeypatch
):
    question_id = "a" * 40
    data_dir = tmp_path / "data/processed/full_validation"
    data_dir.mkdir(parents=True)
    question = {
        "question_id": question_id,
        "paper_id": "p1",
        "paper_title": "Title",
        "question": "What changed?",
        "annotations": [],
    }
    (data_dir / "evaluation_questions.jsonl").write_text(
        json.dumps(question) + "\n", encoding="utf-8"
    )
    for method in ("hybrid", "hybrid_reranked"):
        path = tmp_path / f"results/experiments/full_validation/{method}/evaluation/token-aware"
        path.mkdir(parents=True)
        prediction = {
            "question_id": question_id,
            "retrieved": [{"chunk_id": "c1", "paragraph_id": "p1:p1", "text": "Evidence"}],
        }
        (path / "predictions.jsonl").write_text(
            json.dumps(prediction) + "\n", encoding="utf-8"
        )
    sample = tmp_path / "sample.md"
    sample.write_text(
        f"## Reranker recovered a top-five hit\n### {question_id} — paper `p1`\n",
        encoding="utf-8",
    )
    arguments = [
        "prepare_error_case_review.py", "--root", str(tmp_path),
        "--sample", "sample.md", "--output-dir", "review",
    ]
    monkeypatch.setattr("sys.argv", arguments)
    main()
    output = tmp_path / "review"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_count"] == 1
    assert (output / "review_packet_blinded.md").exists()
    decisions = output / "review_decisions_template.csv"
    decisions.write_text("saved human judgments", encoding="utf-8")
    original_files = {path.name: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr("sys.argv", arguments + ["--seed", "changed-seed"])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original_files
