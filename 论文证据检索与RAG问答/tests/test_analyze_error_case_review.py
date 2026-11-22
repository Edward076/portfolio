import csv
import json

import pytest

from scripts.analyze_error_case_review import analyze_review, main
from scripts.validate_error_case_review import FIELDS


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def review_pair(tmp_path):
    packet, key = tmp_path / "packet.jsonl", tmp_path / "key.jsonl"
    stage1, stage2 = tmp_path / "stage1.csv", tmp_path / "stage2.csv"
    packets, keys, before, after = [], [], [], []
    for index in (1, 2):
        question_id = f"q{index}"
        packets.append({"question_id": question_id, "context_A": [{"chunk_id": "a"}],
                        "context_B": [{"chunk_id": "b"}]})
        keys.append({"question_id": question_id, "outcome_stratum": f"stratum{index}",
                     "method_for_context_A": "hybrid" if index == 1 else "hybrid_reranked",
                     "method_for_context_B": "hybrid_reranked" if index == 1 else "hybrid",
                     "annotations": [{"reference_answers": ["secret gold answer"]}]})
        row = dict.fromkeys(FIELDS, "")
        row.update(question_id=question_id, context_A_sufficiency="sufficient",
                   context_A_relevant_passage_ids="a", context_B_sufficiency="insufficient",
                   pre_unblind_preference="A", confidence_1_to_3="2", reviewer_id="fixture",
                   reviewed_at_utc="2026-10-29T10:00:00Z")
        before.append(row)
        after.append({**row, "post_unblind_preference": "B" if index == 1 else "A",
                      "possible_annotation_gap": "none", "reviewed_at_utc": "2026-10-29T11:00:00Z"})
    packet.write_text("".join(json.dumps(row) + "\n" for row in packets), encoding="utf-8")
    key.write_text("".join(json.dumps(row) + "\n" for row in keys), encoding="utf-8")
    write_csv(stage1, before)
    write_csv(stage2, after)
    return packet, stage1, stage2, key, before, after


def test_method_mapping_respects_case_specific_blinding_and_preserves_inputs(tmp_path):
    packet, stage1, stage2, key, _, _ = review_pair(tmp_path)
    original = {path: path.read_bytes() for path in (packet, stage1, stage2, key)}
    result = analyze_review(packet, stage1, stage2, key)
    assert result["pre_unblind_preference_counts"] == {"hybrid": 1, "hybrid_reranked": 1}
    assert result["post_unblind_preference_counts"] == {"hybrid_reranked": 2}
    assert result["preference_changed_count"] == 1
    assert result["case_count"] == 2 and result["reviewer_count"] == 1
    assert result["by_outcome_stratum"]["stratum1"]["post"] == {"hybrid_reranked": 1}
    assert "secret gold answer" not in json.dumps(result)
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("field,value", [
    ("pre_unblind_preference", "B"), ("reviewer_notes", "edited after unblinding"),
    ("confidence_1_to_3", "3"), ("reviewer_id", "another-reviewer"),
])
def test_changed_stage1_judgments_are_rejected(tmp_path, field, value):
    packet, stage1, stage2, key, _, after = review_pair(tmp_path)
    for row in after:
        row[field] = value
    write_csv(stage2, after)
    with pytest.raises(ValueError, match="stage-1 fields changed"):
        analyze_review(packet, stage1, stage2, key)


def test_incomplete_snapshot_is_rejected_before_key_is_opened(tmp_path):
    packet, stage1, stage2, key, before, _ = review_pair(tmp_path)
    before[0]["pre_unblind_preference"] = ""
    write_csv(stage1, before)
    key.unlink()
    with pytest.raises(ValueError, match="Stage-1 review is incomplete"):
        analyze_review(packet, stage1, stage2, key)


def test_snapshot_with_post_unblinding_fields_is_rejected(tmp_path):
    packet, stage1, stage2, key, before, _ = review_pair(tmp_path)
    before[0]["post_unblind_preference"] = "B"
    write_csv(stage1, before)
    with pytest.raises(ValueError, match="already contains post-unblinding"):
        analyze_review(packet, stage1, stage2, key)


@pytest.mark.parametrize("field,value,error", [
    ("post_unblind_preference", "", "missing stage-2"),
    ("possible_annotation_gap", "", "missing stage-2"),
    ("reviewed_at_utc", "2026-10-29T09:00:00Z", "precedes stage 1"),
])
def test_incomplete_or_earlier_stage2_is_rejected(tmp_path, field, value, error):
    packet, stage1, stage2, key, _, after = review_pair(tmp_path)
    after[0][field] = value
    write_csv(stage2, after)
    with pytest.raises(ValueError, match=error):
        analyze_review(packet, stage1, stage2, key)


def test_invalid_key_cannot_assign_both_labels_to_same_method(tmp_path):
    packet, stage1, stage2, key, _, _ = review_pair(tmp_path)
    rows = [json.loads(line) for line in key.read_text(encoding="utf-8").splitlines()]
    rows[0]["method_for_context_B"] = rows[0]["method_for_context_A"]
    key.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid A/B method mapping"):
        analyze_review(packet, stage1, stage2, key)


def test_cli_does_not_overwrite_a_saved_summary(tmp_path, monkeypatch):
    packet, stage1, stage2, key, _, _ = review_pair(tmp_path)
    output = tmp_path / "summary.json"
    output.write_text("saved summary", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["analyze_error_case_review.py", "--packet", str(packet),
                                    "--stage1", str(stage1), "--decisions", str(stage2),
                                    "--key", str(key), "--output", str(output)])
    with pytest.raises(FileExistsError):
        main()
    assert output.read_text(encoding="utf-8") == "saved summary"


@pytest.mark.parametrize("case", ["missing", "duplicate", "unknown"])
def test_key_question_ids_must_match_exactly(tmp_path, case):
    packet, stage1, stage2, key, _, _ = review_pair(tmp_path)
    rows = [json.loads(line) for line in key.read_text(encoding="utf-8").splitlines()]
    if case == "missing":
        rows.pop()
    elif case == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["question_id"] = "unknown"
    key.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="every packet question exactly once"):
        analyze_review(packet, stage1, stage2, key)
