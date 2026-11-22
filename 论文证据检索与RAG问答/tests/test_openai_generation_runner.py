import json
import sys

import pytest

from scientific_rag.openai_generation import OUTPUT_SCHEMA
from scripts import run_openai_generation as runner
from scripts.run_openai_generation import estimate_cost, load_completed


def test_cost_estimate_includes_each_prompt_and_output_cap():
    jobs = [
        {"messages": [{"role": "user", "content": "one"}]},
        {"messages": [{"role": "user", "content": "two"}]},
    ]

    estimate = estimate_cost(jobs, 0.10, 0.50, max_output_tokens=100)

    assert estimate["request_count"] == 2
    schema_bytes = len(json.dumps(OUTPUT_SCHEMA, separators=(",", ":")).encode("utf-8"))
    assert estimate["estimated_input_tokens"] == (
        len(b'[{"role":"user","content":"one"}]')
        + schema_bytes
        + 2048
        + len(b'[{"role":"user","content":"two"}]')
        + schema_bytes
        + 2048
    )
    assert estimate["max_output_tokens_total"] == 200
    assert estimate["estimated_total_cost_usd"] == pytest.approx(
        estimate["estimated_input_tokens"] * 0.10 / 1_000_000 + 200 * 0.50 / 1_000_000
    )


def test_resume_blocks_uncertain_request_until_usage_is_checked(tmp_path):
    journal = tmp_path / "request_attempts.jsonl"
    journal.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "method": "dense",
                "status": "started",
                "model": "gpt-6-luna",
                "input_price_per_million": 0.10,
                "output_price_per_million": 0.50,
                "max_output_tokens": 256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    jobs = [{"question_id": "q1", "method": "dense"}]

    with pytest.raises(RuntimeError, match="Check provider usage"):
        load_completed(
            tmp_path,
            jobs,
            resume=True,
            model="gpt-6-luna",
            input_price=0.10,
            output_price=0.50,
            max_output_tokens=256,
            retry_uncertain=False,
        )


@pytest.mark.parametrize("invalid_smoke", [False, True])
def test_paired_smoke_can_resume_without_resending_paid_jobs(
    tmp_path, monkeypatch, invalid_smoke,
):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(json.dumps({
            "question_id": question, "paper_id": question, "method": method,
            "messages": [{"role": "user", "content": f"{question} {method}"}],
            "allowed_citations": [], "prompt_version": f"test-{method}",
        }) + "\n" for question in ("q1", "q2") for method in runner.METHODS),
        encoding="utf-8",
    )
    calls = []

    def fake_response(model, messages, api_key, max_output_tokens, *, reasoning_effort):
        calls.append(messages[0]["content"])
        assert reasoning_effort == "none"
        abstained = "true" if invalid_smoke and messages[0]["content"] == "q1 hybrid" else "false"
        return {
            "status": "completed", "id": f"test-{len(calls)}", "model": model,
            "usage": {"input_tokens": 10, "output_tokens": 10},
            "output": [{"type": "message", "content": [{
                "type": "output_text",
                "text": f'{{"answer":"A","abstained":{abstained},"cited_paragraph_ids":[]}}',
            }]}],
        }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    monkeypatch.setattr(runner, "request_response", fake_response)
    output = tmp_path / "run"
    argv = [
        "run_openai_generation", "--execute", "--prompts-file", str(prompts),
        "--output-dir", str(output), "--preflight-output", str(tmp_path / "preflight.json"),
        "--model", "gpt-6-luna", "--reasoning-effort", "none",
        "--input-price-per-million", "0.10", "--output-price-per-million", "0.50",
        "--max-cost-usd", "0.10",
    ]
    monkeypatch.setattr(sys, "argv", [*argv, "--max-requests", "4"])
    runner.main()
    assert calls == [f"q1 {method}" for method in runner.METHODS]

    monkeypatch.setattr(sys, "argv", [*argv, "--resume", "--reasoning-effort", "low"])
    with pytest.raises(ValueError, match="different model or pricing settings"):
        runner.main()
    assert len(calls) == 4

    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    runner.main()
    assert calls == [f"{qid} {method}" for qid in ("q1", "q2") for method in runner.METHODS]
    for method in runner.METHODS:
        rows = runner.read_jsonl(output / f"{method}_predictions.jsonl")
        expected = ["q2"] if invalid_smoke and method == "hybrid" else ["q1", "q2"]
        assert [row["question_id"] for row in rows] == expected
        assert all(row["prompt_version"] == f"test-{method}" for row in rows)
    failures = [
        row for row in runner.read_jsonl(output / "request_attempts.jsonl")
        if row["status"] == "invalid"
    ]
    assert len(failures) == int(invalid_smoke)
    if failures:
        assert failures[0]["cost_usd"] > 0
        assert failures[0]["raw_response"]["status"] == "completed"
