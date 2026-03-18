"""Reciprocal Rank Fusion for combining retrieval results."""

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def rrf_merge(
    *result_lists: list,
    k: int = 60,
    top_k: int | None = None,
) -> list:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across all lists where the document appears.

    Args:
        *result_lists: Variable number of RetrievalResult lists.
        k: RRF constant (default 60, as per original paper).
        top_k: Optional limit on returned results.

    Returns:
        Merged and re-ranked list of RetrievalResult.
    """
    from src.retrieval.dense import RetrievalResult

    scores: dict[str, float] = defaultdict(float)
    best_result: dict[str, Any] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[result.chunk_id] += rrf_score

            if result.chunk_id not in best_result or result.score > best_result[result.chunk_id].score:
                best_result[result.chunk_id] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    if top_k:
        sorted_ids = sorted_ids[:top_k]

    merged = []
    for chunk_id in sorted_ids:
        original = best_result[chunk_id]
        merged.append(RetrievalResult(
            chunk_id=chunk_id,
            score=scores[chunk_id],
            text=original.text,
            metadata=original.metadata,
        ))

    return merged
