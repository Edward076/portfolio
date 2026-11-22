from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


def corpus_fingerprint(chunks: list[dict[str, Any]], model_name: str, model_revision: str) -> str:
    digest = hashlib.sha256(f"{model_name}@{model_revision}".encode())
    for chunk in chunks:
        digest.update(str(chunk["chunk_id"]).encode("utf-8"))
        digest.update(chunk["text"].encode("utf-8"))
    return digest.hexdigest()


class DenseRetriever:
    """CPU-compatible SentenceTransformer embeddings stored in persistent ChromaDB."""

    method_name = "dense"

    def __init__(
        self,
        chunks: Iterable[dict[str, Any]],
        persist_directory: str | Path = "data/cache/chroma",
        model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        model_revision: str = "b207367332321f8e44f96e224ef15bc607f4dbf0",
        batch_size: int = 32,
    ) -> None:
        self.chunks = list(chunks)
        self.model_name = model_name
        self.model_revision = model_revision
        self.batch_size = batch_size
        fingerprint = corpus_fingerprint(self.chunks, model_name, model_revision)
        self.collection_name = f"qasper_{fingerprint[:20]}"
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine", "fingerprint": fingerprint}
        )
        self.model = SentenceTransformer(model_name, revision=model_revision, device="cpu")
        self._ensure_index()

    def _ensure_index(self) -> None:
        indexed = set(self.collection.get(include=[])["ids"])
        missing = [chunk for chunk in self.chunks if chunk["chunk_id"] not in indexed]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self.model.encode(
                [chunk["text"] for chunk in batch],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in batch],
                embeddings=vectors.tolist(),
                documents=[chunk["text"] for chunk in batch],
                metadatas=[
                    {
                        "paper_id": str(chunk["paper_id"]),
                        "paper_title": str(chunk["paper_title"]),
                        "section_name": str(chunk["section_name"]),
                        "paragraph_id": str(chunk["paragraph_id"]),
                    }
                    for chunk in batch
                ],
            )

    def retrieve(self, query: str, paper_id: str, top_k: int = 20) -> list[dict[str, Any]]:
        if not self.chunks or top_k <= 0:
            return []
        vector = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(top_k, len(self.chunks)),
            where={"paper_id": str(paper_id)},
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            ),
            start=1,
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": metadata["paper_id"],
                    "paper_title": metadata["paper_title"],
                    "section_name": metadata["section_name"],
                    "paragraph_id": metadata["paragraph_id"],
                    "text": document,
                    "rank": rank,
                    "score": 1.0 - float(distance),
                    "retriever": "dense",
                }
            )
        return hits
