import csv
import json

import pytest

from scripts.validate_error_case_review import FIELDS, main, validate_review


def review_files(tmp_path, complete=False):
    packet = tmp_path / "packet.jsonl"
    packet.write_text(json.dumps({"question_id": "q1", "context_A": [
        {"chunk_id": "a-child", "paragraph_id": "a-parent"}], "context_B": [
        {"chunk_id": "b-child", "paragraph_id": "b-parent"}]}) + "\n", encoding="utf-8")
    row = dict.fromkeys(FIELDS, "")
    row["question_id"] = "q1"
    if complete:
        row.update(context_A_sufficiency="sufficient", context_B_sufficiency="insufficient",
                   context_A_relevant_passage_ids="a-child", pre_unblind_preference="A",
                   confidence_1_to_3="2", reviewer_id="test-reviewer",
                   reviewed_at_utc="2026-10-29T10:00:00Z")
    decisions = tmp_path / "decisions.csv"
    write_rows(decisions, [row])
    return packet, decisions, row


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_blank_template_is_pending_and_does_not_read_key(tmp_path):
    packet, decisions, _ = review_files(tmp_path)
    (tmp_path / "review_key.jsonl").write_text("INVALID secret answer", encoding="utf-8")
    result = validate_review(packet, decisions)
    assert result["stage1_status"] == "pending"
    assert result["blank_count"] == 1
    assert result["complete_stage1_count"] == result["reviewer_count"] == 0
    assert result["pre_unblind_preference_counts_completed_rows"] == {}
    assert "secret answer" not in json.dumps(result)


def test_completed_fields_report_blinded_counts_only(tmp_path):
    packet, decisions, _ = review_files(tmp_path, complete=True)
    result = validate_review(packet, decisions)
    assert result["stage1_status"] == "complete"
    assert result["complete_stage1_count"] == result["reviewer_count"] == 1
    assert result["pre_unblind_preference_counts_completed_rows"] == {"A": 1}
    assert "not assessed" in result["stage2_review"]
    assert "not verification" in result["scope"]


@pytest.mark.parametrize("field,value,error", [
    ("context_A_relevant_passage_ids", "a-parent", "chunk IDs"),
    ("context_A_relevant_passage_ids", "b-child", "chunk IDs"),
    ("context_A_relevant_passage_ids", "a-child;a-child", "chunk IDs"),
    ("confidence_1_to_3", "4", "confidence_1_to_3"),
    ("pre_unblind_preference", "hybrid", "pre_unblind_preference"),
    ("reviewed_at_utc", "2026-10-29T10:00:00", "UTC timezone"),
    ("reviewed_at_utc", "not-a-time", "invalid reviewed_at_utc"),
])
def test_invalid_record_is_rejected(tmp_path, field, value, error):
    packet, decisions, row = review_files(tmp_path, complete=True)
    row[field] = value
    write_rows(decisions, [row])
    with pytest.raises(ValueError, match=error):
        validate_review(packet, decisions)


def test_uncertain_judgment_needs_notes_and_partial_rows_are_not_votes(tmp_path):
    packet, decisions, row = review_files(tmp_path, complete=True)
    row["context_B_sufficiency"] = "uncertain"
    write_rows(decisions, [row])
    result = validate_review(packet, decisions)
    assert result["stage1_status"] == "partial"
    assert result["pre_unblind_preference_counts_completed_rows"] == {}
    assert "reviewer_notes" in result["cases"][0]["missing_stage1_fields"]


@pytest.mark.parametrize("case", ["duplicate", "missing", "unknown"])
def test_question_set_cannot_change_the_review_denominator(tmp_path, case):
    packet, decisions, row = review_files(tmp_path)
    rows = [row, row] if case == "duplicate" else []
    if case == "unknown":
        rows = [{**row, "question_id": "not-in-packet"}]
    write_rows(decisions, rows)
    with pytest.raises(ValueError, match="every packet question exactly once"):
        validate_review(packet, decisions)


def test_cli_will_not_overwrite_an_existing_report(tmp_path, monkeypatch):
    packet, decisions, _ = review_files(tmp_path)
    output = tmp_path / "report.json"
    output.write_text("existing report", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_error_case_review.py", "--packet", str(packet),
                                    "--decisions", str(decisions), "--output", str(output)])
    with pytest.raises(FileExistsError):
        main()
    assert output.read_text(encoding="utf-8") == "existing report"
