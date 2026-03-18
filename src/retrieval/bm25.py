"""BM25 sparse retrieval with Turkish language support."""

import logging
import pickle
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TURKISH_STOPWORDS = {
    "bir", "bu", "da", "de", "ve", "ile", "için", "olan", "olarak", "gibi",
    "daha", "en", "çok", "her", "kadar", "sonra", "önce", "ise", "ya",
    "ne", "nasıl", "neden", "nerede", "kim", "hangi", "o", "şu", "ben",
    "sen", "biz", "siz", "onlar", "mi", "mu", "mü", "mı", "dir", "dır",
    "dur", "dür", "tir", "tır", "tur", "tür", "ki", "ama", "ancak",
    "fakat", "lakin", "veya", "yahut", "hem", "üzere", "göre", "karşı",
    "arasında", "tarafından", "dolayı", "halde", "rağmen", "itibaren",
    "değil", "var", "yok", "olan", "eden", "eder", "etti", "oldu",
    "olur", "olmuş", "olan", "olan", "iken", "olup", "buna", "şöyle",
    "böyle", "öyle", "aynı", "bazı", "birçok", "diğer", "başka",
}

TURKISH_LOWER_MAP = str.maketrans("İIÖÜÇŞĞ", "iıöüçşğ")


def turkish_tokenize(text: str) -> list[str]:
    """Tokenize Turkish text with proper lowercasing and stopword removal.

    Args:
        text: Input text.

    Returns:
        List of tokens.
    """
    text = text.translate(TURKISH_LOWER_MAP).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in TURKISH_STOPWORDS and len(t) > 1]


def build_bm25_index(
    chunks: list[dict[str, Any]],
    save_path: Path | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build a BM25 index from chunks.

    Args:
        chunks: List of chunk dicts with 'text' and 'chunk_id' keys.
        save_path: Optional path to save index.

    Returns:
        Tuple of (BM25Okapi index, chunk mapping list).
    """
    from rank_bm25 import BM25Okapi

    logger.info(f"Building BM25 index from {len(chunks)} chunks...")
    tokenized = [turkish_tokenize(chunk["text"]) for chunk in chunks]

    index = BM25Okapi(tokenized)

    chunk_mapping = [
        {"chunk_id": c["chunk_id"], "text": c["text"], **{k: v for k, v in c.items() if k not in ("chunk_id", "text")}}
        for c in chunks
    ]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"index": index, "mapping": chunk_mapping}, f)
        logger.info(f"Saved BM25 index to {save_path}")

    return index, chunk_mapping


def load_bm25_index(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    """Load a BM25 index from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        Tuple of (BM25Okapi index, chunk mapping list).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded BM25 index ({len(data['mapping'])} chunks)")
    return data["index"], data["mapping"]


def bm25_search(
    query: str,
    index: Any,
    chunk_mapping: list[dict[str, Any]],
    k: int = 50,
) -> list:
    """Search BM25 index.

    Args:
        query: Query string.
        index: BM25Okapi index.
        chunk_mapping: List of chunk metadata dicts.
        k: Number of results to return.

    Returns:
        List of RetrievalResult sorted by score descending.
    """
    from src.retrieval.dense import RetrievalResult

    tokens = turkish_tokenize(query)
    scores = index.get_scores(tokens)

    top_indices = scores.argsort()[-k:][::-1]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        chunk = chunk_mapping[idx]
        results.append(RetrievalResult(
            chunk_id=chunk["chunk_id"],
            score=float(scores[idx]),
            text=chunk["text"],
            metadata={k_: v for k_, v in chunk.items() if k_ not in ("chunk_id", "text")},
        ))

    return results
