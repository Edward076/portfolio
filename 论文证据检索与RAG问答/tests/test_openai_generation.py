import pytest

from scientific_rag.openai_generation import (
    OUTPUT_SCHEMA,
    build_responses_payload,
    extract_response_text,
)


def test_responses_payload_uses_strict_schema_and_bounded_nonstored_request():
    payload = build_responses_payload(
        "gpt-6-luna",
        [{"role": "user", "content": "Return JSON."}],
        max_output_tokens=128,
        reasoning_effort="none",
    )

    assert payload["model"] == "gpt-6-luna"
    assert payload["max_output_tokens"] == 128
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["store"] is False
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"] == OUTPUT_SCHEMA


def test_extract_response_text_accepts_completed_structured_output():
    response = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"answer":"A","abstained":false,"cited_paragraph_ids":[]}',
                    }
                ],
            }
        ],
    }

    assert extract_response_text(response).startswith('{"answer"')


def test_extract_response_text_rejects_incomplete_and_refusal_responses():
    with pytest.raises(ValueError, match="not completed"):
        extract_response_text({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}})
    with pytest.raises(ValueError, match="refused"):
        extract_response_text(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "Declined"}],
                    }
                ],
            }
        )
