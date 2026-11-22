from __future__ import annotations

from typing import Any


def split_token_aware(
    chunks: list[dict[str, Any]], tokenizer, max_tokens: int = 450, overlap_tokens: int = 50
) -> list[dict[str, Any]]:
    """Split long paragraphs into overlapping subchunks with stable parent IDs."""
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Require max_tokens > overlap_tokens >= 0")
    result: list[dict[str, Any]] = []
    stride = max_tokens - overlap_tokens
    for chunk in chunks:
        token_ids = tokenizer.encode(chunk["text"], add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            result.append(chunk)
            continue
        for part_index, start in enumerate(range(0, len(token_ids), stride)):
            part_ids = token_ids[start : start + max_tokens]
            if not part_ids:
                continue
            text = tokenizer.decode(part_ids, skip_special_tokens=True).strip()
            if not text:
                continue
            result.append(
                {
                    **chunk,
                    "chunk_id": f"{chunk['paragraph_id']}:c{part_index:03d}",
                    "text": text,
                    "subchunk_index": part_index,
                    "parent_chunk_id": chunk["chunk_id"],
                    "token_start": start,
                    "token_end": min(start + max_tokens, len(token_ids)),
                }
            )
            if start + max_tokens >= len(token_ids):
                break
    return result
