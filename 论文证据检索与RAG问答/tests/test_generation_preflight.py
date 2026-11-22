from scripts.prepare_generation_prompts import (
    answerability_stratum,
    select_balanced_questions,
)


def test_answerability_strata_are_disjoint_and_use_mapping_status():
    assert answerability_stratum({"annotations": []}) == "no_annotations"
    assert answerability_stratum(
        {"annotations": [{"unanswerable": True, "mapped_evidence_ids": []}]}
    ) == "fully_unanswerable"
    assert answerability_stratum(
        {
            "annotations": [
                {"unanswerable": True, "mapped_evidence_ids": []},
                {"unanswerable": False, "mapped_evidence_ids": ["p1:p1"]},
            ]
        }
    ) == "mixed_answerability"
    assert answerability_stratum(
        {"annotations": [{"unanswerable": False, "mapped_evidence_ids": ["p1:p1"]}]}
    ) == "answerable_mapped_evidence"
    assert answerability_stratum(
        {"annotations": [{"unanswerable": False, "mapped_evidence_ids": []}]}
    ) == "answerable_unmapped_evidence"


def test_balanced_question_selection_is_reproducible():
    questions = [
        {
            "question_id": f"q{index}",
            "annotations": [{"unanswerable": index % 2 == 0, "mapped_evidence_ids": []}],
        }
        for index in range(8)
    ]

    first, counts = select_balanced_questions(questions, per_stratum=2, seed="42")
    second, _ = select_balanced_questions(list(reversed(questions)), per_stratum=2, seed="42")

    assert counts == {"answerable_unmapped_evidence": 2, "fully_unanswerable": 2}
    assert [(row["question_id"], row["_sampling_stratum"]) for row in first] == [
        (row["question_id"], row["_sampling_stratum"]) for row in second
    ]
