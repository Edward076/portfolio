from __future__ import annotations

import time
from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]], rrf_constant: int = 60, top_k: int = 20
) -> list[dict[str, Any]]:
    if rrf_constant < 0 or top_k < 0:
        raise ValueError("rrf_constant and top_k must be nonnegative")
    fused: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for position, item in enumerate(ranked, start=1):
            chunk_id = str(item["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id not in fused:
                fused[chunk_id] = {**item, "rrf_score": 0.0, "source_ranks": []}
            fused[chunk_id]["rrf_score"] += 1.0 / (rrf_constant + position)
            fused[chunk_id]["source_ranks"].append(position)
    result = sorted(fused.values(), key=lambda item: (-item["rrf_score"], item["chunk_id"]))[:top_k]
    for rank, item in enumerate(result, start=1):
        item["rank"] = rank
        item["score"] = item["rrf_score"]
        item["retriever"] = "hybrid"
    return result


class HybridRetriever:
    method_name = "hybrid"

    def __init__(
        self, dense, bm25, dense_top_k: int = 20, bm25_top_k: int = 20, rrf_constant: int = 60
    ):
        self.dense = dense
        self.bm25 = bm25
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_constant = rrf_constant
        self.last_component_latency_ms: dict[str, float] = {}

    def retrieve(self, query: str, paper_id: str, top_k: int = 20) -> list[dict[str, Any]]:
        started = time.perf_counter()
        dense_hits = self.dense.retrieve(query, paper_id, self.dense_top_k)
        dense_latency = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        bm25_hits = self.bm25.retrieve(query, paper_id, self.bm25_top_k)
        bm25_latency = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        fused = reciprocal_rank_fusion([dense_hits, bm25_hits], self.rrf_constant, top_k)
        fusion_latency = (time.perf_counter() - started) * 1000
        self.last_component_latency_ms = {
            "dense_retrieval_latency_ms": dense_latency,
            "bm25_retrieval_latency_ms": bm25_latency,
            "rrf_fusion_latency_ms": fusion_latency,
        }
        return fused
