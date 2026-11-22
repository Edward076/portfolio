"""Run the prepared Qasper pilot through a llama.cpp server on loopback."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from scientific_rag.generation_prompt import parse_generation_response
from scientific_rag.openai_generation import OUTPUT_SCHEMA

METHODS = ("closed_book", "dense", "hybrid", "hybrid_reranked")
DECODE = {"temperature": 0, "seed": 42, "max_tokens": 256, "cache_prompt": False}


def request(port: int, route: str, data: dict | None = None) -> tuple[int, str]:
    # HTTPConnection connects directly to IPv4 loopback and never follows redirects.
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        if data is None:
            connection.request("GET", route)
        else:
            connection.request(
                "POST", route, json.dumps(data).encode("utf-8"),
                {"Content-Type": "application/json"},
            )
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


def payload(job: dict, model: str) -> dict:
    citations = job["allowed_citations"]
    citation_schema = (
        {"type": "array", "items": {"enum": citations}, "maxItems": len(citations)}
        if citations else {"const": []}
    )
    # The grammar constrains the response shape and citation IDs, not answer correctness.
    answer_schema = {
        **OUTPUT_SCHEMA,
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "abstained": {"const": False},
            "cited_paragraph_ids": citation_schema,
        },
    }
    abstain_schema = {
        **OUTPUT_SCHEMA,
        "properties": {
            "answer": {"const": ""},
            "abstained": {"const": True},
            "cited_paragraph_ids": {"const": []},
        },
    }
    return {
        **DECODE, "model": model, "messages": job["messages"], "stream": False,
        "response_format": {
            "type": "json_object", "schema": {"oneOf": [answer_schema, abstain_schema]},
        },
    }


def load_jobs(prompt_file: Path, manifest_file: Path) -> tuple[list[dict], str, str]:
    prompt_bytes = prompt_file.read_bytes()
    manifest_bytes = manifest_file.read_bytes()
    jobs = [json.loads(line) for line in prompt_bytes.splitlines() if line.strip()]
    manifest = json.loads(manifest_bytes)
    selected = manifest["selected_question_ids_by_stratum"]
    question_ids = {qid for ids in selected.values() for qid in ids}
    pairs = Counter((job["question_id"], job["method"]) for job in jobs)
    expected = {(qid, method) for qid in question_ids for method in METHODS}
    if len(question_ids) != 20 or pairs != Counter(expected):
        raise ValueError("Prompt file must contain the same 20 questions in all four arms")
    return (
        jobs, hashlib.sha256(prompt_bytes).hexdigest(),
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def save_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run(args: argparse.Namespace) -> int:
    jobs, prompts_hash, selection_hash = load_jobs(args.prompts, args.selection)
    runtime = json.loads(args.runtime_metadata.read_text(encoding="utf-8-sig"))
    status, models = request(args.port, "/v1/models")
    if status != 200 or args.model not in {row["id"] for row in json.loads(models)["data"]}:
        raise ValueError("The requested model is not loaded on the local server")
    status, props = request(args.port, "/props")
    if status != 200:
        raise ValueError("Cannot verify local server properties")
    server_props = json.loads(props)
    context_limit = server_props["default_generation_settings"]["n_ctx"]
    if args.smoke:
        jobs = [job for job in jobs if job["question_id"] == jobs[0]["question_id"]]

    args.output_dir.mkdir(parents=True)
    save_json(args.output_dir / "run_manifest.json", {
        "started_at_utc": datetime.now(UTC).isoformat(), "model": args.model,
        "runtime": runtime, "server_properties": server_props, "context_limit": context_limit,
        "prompt_file": str(args.prompts), "prompt_sha256": prompts_hash,
        "selection_sha256": selection_hash, "smoke": args.smoke, "jobs": len(jobs),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "question_ids": list(dict.fromkeys(job["question_id"] for job in jobs)),
        "decoding": DECODE, "external_provider_calls": 0,
        "citation_constraint": (
            "Only listed context IDs, at most one entry per available ID in total; "
            "closed-book citations must be empty"
        ),
        "output_constraint": "Schema requires answer/abstention consistency",
    })
    predictions = {method: [] for method in METHODS}
    with (args.output_dir / "attempts.jsonl").open("x", encoding="utf-8") as log:
        for index, job in enumerate(jobs, 1):
            entry = {
                "question_id": job["question_id"], "method": job["method"],
                "started_at_utc": datetime.now(UTC).isoformat(),
            }
            started = perf_counter()
            try:
                status, raw = request(args.port, "/v1/chat/completions", payload(job, args.model))
                entry.update(http_status=status, raw_reply=raw)
                if status != 200:
                    raise ValueError(f"Server returned HTTP {status}")
                response = json.loads(raw)
                choice, = response["choices"]
                usage = response["usage"]
                if choice["finish_reason"] != "stop":
                    raise ValueError(f"Generation ended with {choice['finish_reason']!r}")
                if usage["prompt_tokens"] + DECODE["max_tokens"] > context_limit:
                    raise ValueError("Prompt and output budget exceed verified context limit")
                prediction = parse_generation_response(
                    choice["message"]["content"], job["allowed_citations"],
                )
                prediction.update(
                    question_id=job["question_id"], method=job["method"], model=args.model,
                    prompt_version=job["prompt_version"], input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                    latency_ms=round((perf_counter() - started) * 1000, 3),
                )
                predictions[job["method"]].append(prediction)
                entry["status"] = "valid"
            except (OSError, ValueError, TypeError, KeyError, http.client.HTTPException) as exc:
                entry.update(status="invalid", error=str(exc))
            entry["latency_ms"] = round((perf_counter() - started) * 1000, 3)
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{index}/{len(jobs)} {job['method']} {entry['status']}", flush=True)

    valid = sum(map(len, predictions.values()))
    if valid == len(jobs):
        for method, rows in predictions.items():
            with (args.output_dir / f"{method}_predictions.jsonl").open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(args.output_dir / "run_status.json", {
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "attempted": len(jobs), "valid": valid, "invalid": len(jobs) - valid,
        "paired_outputs_written": valid == len(jobs),
    })
    return 0 if valid == len(jobs) else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    preflight = Path("results/experiments/generation_pilot_preflight")
    parser.add_argument("--prompts", type=Path, default=preflight / "prompts.jsonl")
    parser.add_argument("--selection", type=Path, default=preflight / "preflight_manifest.json")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
