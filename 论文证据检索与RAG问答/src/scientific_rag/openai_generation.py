"""Small OpenAI Responses API adapter; never invoked by offline evaluation code."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "abstained": {"type": "boolean"},
        "cited_paragraph_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "abstained", "cited_paragraph_ids"],
    "additionalProperties": False,
}


def build_responses_payload(
    model: str, messages: Sequence[Mapping[str, str]], max_output_tokens: int,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("model must be nonempty")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    payload = {
        "model": model,
        "input": list(messages),
        "max_output_tokens": max_output_tokens,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "qasper_grounded_answer",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    }
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def request_response(
    model: str,
    messages: Sequence[Mapping[str, str]],
    api_key: str,
    max_output_tokens: int,
    timeout_seconds: int = 120,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Send exactly one request. Errors are raised and never retried automatically."""
    if not api_key.strip():
        raise ValueError("API key must be nonempty")
    payload = build_responses_payload(model, messages, max_output_tokens, reasoning_effort)
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI Responses API returned HTTP {exc.code}; no retry attempted") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"OpenAI Responses API connection failed ({exc.reason}); no retry attempted"
        ) from None
    if not isinstance(result, dict):
        raise TypeError("OpenAI response must be a JSON object")
    return result


def extract_response_text(response: Mapping[str, Any]) -> str:
    if response.get("status") != "completed":
        detail = response.get("incomplete_details")
        raise ValueError(f"response was not completed: {detail!r}")
    pieces = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise ValueError("model refused the request; response was not scored")
            if content.get("type") == "output_text":
                pieces.append(content.get("text", ""))
    if not pieces:
        raise ValueError("completed response contains no output text")
    return "".join(pieces)
