from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from scientific_rag.generation_prompt import parse_generation_response
from scientific_rag.openai_generation import OUTPUT_SCHEMA, extract_response_text, request_response

METHODS = ("closed_book", "dense", "hybrid", "hybrid_reranked")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompt_token_estimate(job: dict[str, Any]) -> int:
    """Byte-based planning estimate, including schema and fixed request overhead."""
    serialized = json.dumps(job["messages"], ensure_ascii=False, separators=(",", ":"))
    schema = json.dumps(OUTPUT_SCHEMA, separators=(",", ":"))
    # One token per UTF-8 byte is intentionally conservative for ordinary BPE text.
    # The fixed reserve covers roles, formatting, and API-side prompt framing.
    return len(serialized.encode("utf-8")) + len(schema.encode("utf-8")) + 2048


def estimate_cost(
    jobs: list[dict[str, Any]],
    input_price_per_million: float,
    output_price_per_million: float,
    max_output_tokens: int,
) -> dict[str, float | int]:
    estimated_input_tokens = sum(prompt_token_estimate(job) for job in jobs)
    estimated_output_tokens = len(jobs) * max_output_tokens
    input_cost = estimated_input_tokens * input_price_per_million / 1_000_000
    output_cost = estimated_output_tokens * output_price_per_million / 1_000_000
    return {
        "request_count": len(jobs),
        "estimated_input_tokens": estimated_input_tokens,
        "max_output_tokens_total": estimated_output_tokens,
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_total_cost_usd": input_cost + output_cost,
    }


def load_completed(
    output_dir: Path,
    jobs: list[dict[str, Any]],
    resume: bool,
    model: str,
    input_price: float,
    output_price: float,
    max_output_tokens: int,
    retry_uncertain: bool,
    reasoning_effort: str | None = None,
) -> tuple[set[tuple[str, str]], float]:
    """Load finished requests and accounted cost, including invalid replies."""
    expected = {(job["question_id"], job["method"]) for job in jobs}
    completed: set[tuple[str, str]] = set()
    prior_cost = 0.0
    for method in METHODS:
        path = output_dir / f"{method}_predictions.jsonl"
        if not path.exists():
            continue
        if not resume:
            raise FileExistsError(f"{path} already exists; pass --resume to continue")
        for row in read_jsonl(path):
            key = (row["question_id"], row["method"])
            if key not in expected:
                raise ValueError(f"Unexpected existing generation record: {key}")
            if key in completed:
                raise ValueError(f"Duplicate existing generation record: {key}")
            if "cost_usd" not in row:
                raise ValueError(f"Existing record has no cost ledger: {key}")
            prior_config = (
                row.get("model"),
                float(row.get("input_price_per_million", -1)),
                float(row.get("output_price_per_million", -1)),
                int(row.get("max_output_tokens", -1)),
                row.get("reasoning_effort"),
            )
            expected_config = (model, input_price, output_price, max_output_tokens, reasoning_effort)
            if prior_config != expected_config:
                raise ValueError(
                    f"Existing job {key} uses different model or pricing settings; "
                    "resume in a new output directory"
                )
            completed.add(key)
            prior_cost += float(row["cost_usd"])

    attempts_path = output_dir / "request_attempts.jsonl"
    latest_attempt: dict[tuple[str, str], dict[str, Any]] = {}
    if attempts_path.exists():
        if not resume:
            raise FileExistsError(f"{attempts_path} already exists; pass --resume to continue")
        for row in read_jsonl(attempts_path):
            key = (row["question_id"], row["method"])
            if key not in expected:
                raise ValueError(f"Unexpected request attempt in journal: {key}")
            latest_attempt[key] = row
    uncertain = {
        key
        for key, row in latest_attempt.items()
        if row["status"] == "started" and key not in completed
    }
    if uncertain and not retry_uncertain:
        raise RuntimeError(
            "Some requests started but have no saved response. Check provider usage before "
            "retrying; no automatic resend was made. If you verified they were not charged, "
            "rerun with --resume --retry-uncertain."
        )
    for key, row in latest_attempt.items():
        if key not in uncertain and row["status"] != "invalid":
            continue
        prior_config = (
            row.get("model"),
            float(row.get("input_price_per_million", -1)),
            float(row.get("output_price_per_million", -1)),
            int(row.get("max_output_tokens", -1)),
            row.get("reasoning_effort"),
        )
        if prior_config != (model, input_price, output_price, max_output_tokens, reasoning_effort):
            raise ValueError(
                f"Recorded request {key} uses different model or pricing settings; "
                "reconcile it before changing settings"
            )
        if row["status"] == "invalid":
            cost = row.get("cost_usd", row.get("reserved_cost_usd"))
            if key in completed or cost is None or not math.isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid reply {key} has conflicting or missing accounting")
            prior_cost += cost
            completed.add(key)
    return completed, prior_cost


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight or explicitly run the prepared OpenAI generation pilot."
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path("results/experiments/generation_pilot_preflight/prompts.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/experiments/generation_pilot")
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=Path("results/experiments/generation_pilot_preflight/cost_estimate.json"),
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-6-luna"))
    parser.add_argument("--input-price-per-million", type=float, required=True)
    parser.add_argument("--output-price-per-million", type=float, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument(
        "--max-requests", type=int, help="limit this invocation, then continue with --resume",
    )
    parser.add_argument("--execute", action="store_true", help="send paid requests; default is preflight only")
    parser.add_argument("--resume", action="store_true", help="continue from completed output records")
    parser.add_argument(
        "--retry-uncertain",
        action="store_true",
        help="retry only after checking provider usage and confirming an interrupted request was not charged",
    )
    args = parser.parse_args()
    prices = (args.input_price_per_million, args.output_price_per_million)
    if any(not math.isfinite(price) or price < 0 for price in prices):
        parser.error("prices must be finite and nonnegative")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    if args.max_cost_usd is not None and (
        not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0
    ):
        parser.error("--max-cost-usd must be finite and positive")
    if args.execute and args.max_cost_usd is None:
        parser.error("--execute requires an explicit --max-cost-usd")
    if args.retry_uncertain and not (args.execute and args.resume):
        parser.error("--retry-uncertain requires both --execute and --resume")

    jobs = read_jsonl(args.prompts_file)
    if not jobs:
        parser.error("prompt batch is empty")
    keys = [(job.get("question_id"), job.get("method")) for job in jobs]
    if any(not question_id or method not in METHODS for question_id, method in keys):
        parser.error("prompt batch contains an invalid question_id or method")
    if len(set(keys)) != len(keys):
        parser.error("prompt batch contains duplicate question/method jobs")

    completed: set[tuple[str, str]] = set()
    prior_cost = 0.0
    if args.execute:
        completed, prior_cost = load_completed(
            args.output_dir,
            jobs,
            args.resume,
            args.model,
            args.input_price_per_million,
            args.output_price_per_million,
            args.max_output_tokens,
            args.retry_uncertain,
            args.reasoning_effort,
        )
    pending = [job for job in jobs if (job["question_id"], job["method"]) not in completed]
    selected = pending[:args.max_requests]
    estimate = estimate_cost(
        pending,
        args.input_price_per_million,
        args.output_price_per_million,
        args.max_output_tokens,
    )
    projected_total = prior_cost + float(estimate["estimated_total_cost_usd"])
    preflight = {
        "mode": "execute" if args.execute else "dry_run",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prompts_file": str(args.prompts_file),
        "prompts_sha256": hashlib.sha256(args.prompts_file.read_bytes()).hexdigest(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_count": len(jobs),
        "completed_count": len(completed),
        "pending_count": len(pending),
        "requests_this_invocation": len(selected),
        "this_invocation_estimated_cost_usd": estimate_cost(
            selected, args.input_price_per_million, args.output_price_per_million,
            args.max_output_tokens,
        )["estimated_total_cost_usd"],
        **estimate,
        "prior_actual_cost_usd": prior_cost,
        "projected_total_cost_usd": projected_total,
        "max_cost_usd": args.max_cost_usd,
    }
    print(json.dumps(preflight, indent=2))
    args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_output.write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved preflight record: {args.preflight_output}")
    if args.max_cost_usd is not None and projected_total > args.max_cost_usd:
        raise SystemExit("Preflight exceeds --max-cost-usd; no requests were sent")
    if not args.execute:
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set; no requests were sent")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    streams = {}
    attempts_path = args.output_dir / "request_attempts.jsonl"
    attempts_stream = attempts_path.open("a", encoding="utf-8", newline="\n")
    try:
        for method in METHODS:
            method_jobs = [job for job in selected if job["method"] == method]
            if not method_jobs:
                continue
            path = args.output_dir / f"{method}_predictions.jsonl"
            streams[method] = path.open("a", encoding="utf-8", newline="\n")

        actual_run_cost = 0.0
        for index, job in enumerate(selected, start=1):
            next_request_estimate = float(
                estimate_cost(
                    [job],
                    args.input_price_per_million,
                    args.output_price_per_million,
                    args.max_output_tokens,
                )["estimated_total_cost_usd"]
            )
            projected_if_sent = prior_cost + actual_run_cost + next_request_estimate
            if projected_if_sent > args.max_cost_usd:
                raise SystemExit(
                    "Budget guard stopped before sending the next request: its estimated "
                    "cost would exceed --max-cost-usd"
                )
            attempt = {
                "question_id": job["question_id"],
                "method": job["method"],
                "status": "started",
                "model": args.model,
                "input_price_per_million": args.input_price_per_million,
                "output_price_per_million": args.output_price_per_million,
                "max_output_tokens": args.max_output_tokens,
                "reasoning_effort": args.reasoning_effort,
            }
            attempts_stream.write(json.dumps(attempt) + "\n")
            attempts_stream.flush()
            os.fsync(attempts_stream.fileno())
            start = perf_counter()
            response = request_response(
                args.model,
                job["messages"],
                api_key,
                args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            latency_ms = (perf_counter() - start) * 1000
            usage = response.get("usage") or {}
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            if not input_tokens and not output_tokens:
                raise ValueError("API response omitted token usage; stopping without retry")
            cost = (
                input_tokens * args.input_price_per_million
                + output_tokens * args.output_price_per_million
            ) / 1_000_000
            receipt = {
                **attempt,
                "status": "completed",
                "api_model": response.get("model"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
                "latency_ms": latency_ms,
                "cost_usd": cost,
                "response_id": response.get("id"),
            }
            try:
                parsed = parse_generation_response(
                    extract_response_text(response), job["allowed_citations"]
                )
            except (ValueError, TypeError) as exc:
                receipt.update(status="invalid", error=str(exc), raw_response=response)
            else:
                row = {
                    **receipt,
                    **parsed,
                    "paper_id": job["paper_id"],
                    "prompt_version": job.get("prompt_version", parsed["prompt_version"]),
                }
                stream = streams[job["method"]]
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            attempts_stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
            attempts_stream.flush()
            os.fsync(attempts_stream.fileno())
            actual_run_cost += cost
            print(
                json.dumps(
                    {
                        "accounted": index,
                        "status": receipt["status"],
                        "planned_this_invocation": len(selected),
                        "method": job["method"],
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "actual_run_cost_usd": actual_run_cost,
                    }
                )
            )
            if prior_cost + actual_run_cost > args.max_cost_usd:
                raise SystemExit(
                    "Observed provider usage exceeded --max-cost-usd after this response; "
                    "the response was checkpointed and no further requests will be sent"
                )
    finally:
        for stream in streams.values():
            stream.close()
        attempts_stream.close()


if __name__ == "__main__":
    main()
