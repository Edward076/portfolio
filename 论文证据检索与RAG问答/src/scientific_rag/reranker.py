from __future__ import annotations

import time
from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        model_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, revision=model_revision, device="cpu")

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if not candidates or top_k <= 0:
            return []
        scores = self.model.predict(
            [(query, item["text"]) for item in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0]["chunk_id"]),
        )[:top_k]
        return [
            {
                **item,
                "reranker_score": float(score),
                "pre_rerank_rank": item["rank"],
                "rank": rank,
                "retriever": "hybrid_reranked",
            }
            for rank, (item, score) in enumerate(ranked, start=1)
        ]


class RerankedHybridRetriever:
    method_name = "hybrid_reranked"

    def __init__(self, hybrid, reranker: CrossEncoderReranker, candidate_top_k: int = 20):
        self.hybrid = hybrid
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k
        self.last_component_latency_ms: dict[str, float] = {}

    def retrieve(self, query: str, paper_id: str, top_k: int = 20) -> list[dict[str, Any]]:
        hybrid_started = time.perf_counter()
        candidates = self.hybrid.retrieve(query, paper_id, self.candidate_top_k)
        hybrid_latency = (time.perf_counter() - hybrid_started) * 1000
        component_latencies = dict(getattr(self.hybrid, "last_component_latency_ms", {}))
        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(query, candidates, top_k=len(candidates))
        rerank_latency = (time.perf_counter() - rerank_started) * 1000
        self.last_component_latency_ms = {
            **component_latencies,
            "hybrid_candidate_latency_ms": hybrid_latency,
            "reranking_latency_ms": rerank_latency,
        }
        return reranked
