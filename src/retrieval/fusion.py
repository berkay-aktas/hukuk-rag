"""Reciprocal Rank Fusion (RRF) for combining retrieval results.

Implements the method from Cormack, Clarke & Buettcher (2009):
``score(d) = Σ 1 / (k + rank_i(d))`` across all input rankings.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


def rrf_merge(
    *result_lists: list[RetrievalResult],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        *result_lists: Variable number of RetrievalResult lists.
        k: RRF constant (default 60, as per original paper).
        top_k: Optional limit on returned results. None returns all.

    Returns:
        Merged and re-ranked list of RetrievalResult.
    """
    scores: dict[str, float] = defaultdict(float)
    best_result: dict[str, RetrievalResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[result.chunk_id] += rrf_score

            if result.chunk_id not in best_result or result.score > best_result[result.chunk_id].score:
                best_result[result.chunk_id] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    return [
        RetrievalResult(
            chunk_id=chunk_id,
            score=scores[chunk_id],
            text=best_result[chunk_id].text,
            metadata=best_result[chunk_id].metadata,
        )
        for chunk_id in sorted_ids
    ]
