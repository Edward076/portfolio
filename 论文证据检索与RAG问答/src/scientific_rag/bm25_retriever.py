from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rank_bm25 import BM25Okapi

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    """Paper-scoped BM25 index over paragraph chunks."""

    method_name = "bm25"

    def __init__(self, chunks: Iterable[dict[str, Any]]) -> None:
        self.chunks_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            self.chunks_by_paper[str(chunk["paper_id"])].append(chunk)
        self.indices: dict[str, BM25Okapi] = {}
        for paper_id, paper_chunks in self.chunks_by_paper.items():
            self.indices[paper_id] = BM25Okapi([tokenize(chunk["text"]) for chunk in paper_chunks])

    def retrieve(self, query: str, paper_id: str, top_k: int = 20) -> list[dict[str, Any]]:
        chunks = self.chunks_by_paper.get(str(paper_id), [])
        index = self.indices.get(str(paper_id))
        if not chunks or index is None or top_k <= 0:
            return []
        scores = index.get_scores(tokenize(query))
        ranked = sorted(
            range(len(chunks)), key=lambda position: (-float(scores[position]), position)
        )[:top_k]
        return [
            {
                **chunks[position],
                "rank": rank,
                "score": float(scores[position]),
                "retriever": "bm25",
            }
            for rank, position in enumerate(ranked, start=1)
        ]
