from scripts.compare_runtime_outputs import compare_predictions


def make_prediction(question_id="q1", *, text="Evidence text.", score=0.9):
    return {
        "question_id": question_id,
        "gold_evidence_sets": [["paper:p001-000"]],
        "retrieved": [
            {
                "chunk_id": "paper:p001-000:c000",
                "paragraph_id": "paper:p001-000",
                "text": text,
                "score": score,
            }
        ],
    }


def test_runtime_comparison_accepts_identical_ranked_contexts():
    result = compare_predictions([make_prediction()], [make_prediction()])

    assert result["same_question_ids"] is True
    assert result["ranked_parent_paragraph_ids_match"] is True
    assert result["ranked_chunk_ids_match"] is True
    assert result["ranked_text_match"] is True
    assert result["retrieval_scores_match"] is True
    assert result["gold_evidence_sets_match"] is True


def test_runtime_comparison_detects_context_and_question_id_changes():
    result = compare_predictions(
        [make_prediction()],
        [make_prediction(text="Changed text.", score=0.8)],
    )

    assert result["same_question_ids"] is True
    assert result["ranked_text_match"] is False
    assert result["retrieval_scores_match"] is False
    assert result["mismatched_question_counts"]["ranked_text_match"] == 1


def test_runtime_comparison_detects_question_id_changes():
    result = compare_predictions([make_prediction()], [make_prediction(question_id="q2")])

    assert result["same_question_ids"] is False
    assert result["shared_question_count"] == 0
