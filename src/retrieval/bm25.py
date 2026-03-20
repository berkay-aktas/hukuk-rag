"""BM25 sparse retrieval with Turkish language support."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.retrieval.types import RetrievalResult
from src.utils.turkish import turkish_tokenize

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def build_bm25_index(
    chunks: list[dict[str, Any]],
    save_path: Path | None = None,
) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Build a BM25 index from chunks.

    Uses Turkish-aware tokenization with proper locale lowercasing
    (İ→i, I→ı) and stopword removal.

    Args:
        chunks: List of chunk dicts with 'text' and 'chunk_id' keys.
        save_path: Optional path to save index as pickle.

    Returns:
        Tuple of (BM25Okapi index, chunk mapping list).
    """
    from rank_bm25 import BM25Okapi

    logger.info(f"Building BM25 index from {len(chunks)} chunks...")
    tokenized = [turkish_tokenize(chunk["text"]) for chunk in chunks]

    index = BM25Okapi(tokenized)

    chunk_mapping = [
        {"chunk_id": c["chunk_id"], "text": c["text"],
         **{key: val for key, val in c.items() if key not in ("chunk_id", "text")}}
        for c in chunks
    ]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"index": index, "mapping": chunk_mapping}, f)
        logger.info(f"Saved BM25 index to {save_path}")

    return index, chunk_mapping


def load_bm25_index(path: Path) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Load a BM25 index from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        Tuple of (BM25Okapi index, chunk mapping list).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    logger.info(f"Loaded BM25 index ({len(data['mapping'])} chunks)")
    return data["index"], data["mapping"]


def bm25_search(
    query: str,
    index: BM25Okapi,
    chunk_mapping: list[dict[str, Any]],
    k: int = 50,
) -> list[RetrievalResult]:
    """Search BM25 index with Turkish-tokenized query.

    Args:
        query: Query string.
        index: BM25Okapi index.
        chunk_mapping: List of chunk metadata dicts.
        k: Number of results to return.

    Returns:
        List of RetrievalResult sorted by score descending.
    """
    import numpy as np

    tokens = turkish_tokenize(query)
    scores = index.get_scores(tokens)

    top_indices = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        chunk = chunk_mapping[idx]
        results.append(RetrievalResult(
            chunk_id=chunk["chunk_id"],
            score=float(scores[idx]),
            text=chunk["text"],
            metadata={key: val for key, val in chunk.items() if key not in ("chunk_id", "text")},
        ))

    return results
