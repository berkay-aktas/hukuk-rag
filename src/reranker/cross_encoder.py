"""Cross-encoder reranker: scores (query, passage) pairs jointly.

Default model is :data:`DEFAULT_RERANKER` — the off-the-shelf Turkish-tuned
bge-reranker-v2-m3. Pass ``model_path`` to a Drive checkpoint to use a
fine-tuned variant.

Note on the fine-tuned variant from prior sessions: it was trained on silver
labels (52% of training queries had no true positive) and consistently
degraded downstream Token F1 by 11% at p<0.001. Use the off-the-shelf model
in production unless you've retrained on clean labels (e.g. the shared
``reranker.jsonl`` with ``audit_status: no_eval_gold_leak_balanced_v3``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.retrieval.types import RetrievalResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "seroe/bge-reranker-v2-m3-turkish-triplet"


@dataclass
class LoadedReranker:
    """A loaded cross-encoder model."""

    model: CrossEncoder
    model_id: str


def load_reranker(
    model_path: str | Path = DEFAULT_RERANKER,
    *,
    device: str | None = None,
    max_length: int = 512,
) -> LoadedReranker:
    """Load a cross-encoder reranker.

    Args:
        model_path: HuggingFace model id or local path to a fine-tuned checkpoint.
        device: ``"cuda"``, ``"mps"``, ``"cpu"``, or None for auto-select.
        max_length: Truncate (query + passage) to this many tokens.

    Returns:
        A :class:`LoadedReranker`.
    """
    from sentence_transformers import CrossEncoder

    if device is None:
        device = _auto_device()
    logger.info("Loading cross-encoder reranker: %s (device=%s)", model_path, device)
    model = CrossEncoder(str(model_path), max_length=max_length, device=device)
    return LoadedReranker(model=model, model_id=str(model_path))


def rerank(
    reranker: LoadedReranker,
    query: str,
    candidates: list[RetrievalResult],
    *,
    top_k: int | None = None,
    batch_size: int = 32,
) -> list[RetrievalResult]:
    """Re-score and re-order candidates by cross-encoder relevance.

    Args:
        reranker: A :class:`LoadedReranker`.
        query: The query string.
        candidates: First-stage retrieval results to rerank.
        top_k: Optionally truncate to top-k after reranking.
        batch_size: Forward-pass batch size for the cross-encoder.

    Returns:
        Candidates sorted by cross-encoder score descending. Each result's
        ``score`` is replaced with the rerank score (original retrieval score
        is preserved in ``metadata["first_stage_score"]``).
    """
    if not candidates:
        return []

    # Cross-encoder input format is [[query, passage], ...] — not [(q, p)] tuples.
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    reranked = []
    for c, s in zip(candidates, scores):
        meta = dict(c.metadata) if c.metadata else {}
        meta["first_stage_score"] = c.score
        reranked.append(RetrievalResult(
            chunk_id=c.chunk_id,
            score=float(s),
            text=c.text,
            metadata=meta,
        ))
    reranked.sort(key=lambda r: r.score, reverse=True)
    if top_k is not None:
        reranked = reranked[:top_k]
    return reranked


def _auto_device() -> str:
    """Pick the best available device for the reranker."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
