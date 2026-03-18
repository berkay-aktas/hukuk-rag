"""Dense retrieval with FAISS and sentence-transformers."""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


def build_faiss_index(
    chunks: list[dict[str, Any]],
    model_name: str = "intfloat/multilingual-e5-large",
    save_path: Path | None = None,
    batch_size: int = 64,
    nlist: int = 256,
    m: int = 32,
    nbits: int = 8,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build a FAISS IVF-PQ index from chunks.

    Args:
        chunks: List of chunk dicts with 'text' and 'chunk_id' keys.
        model_name: Sentence-transformer model to use for embedding.
        save_path: Optional path to save index and mapping.
        batch_size: Batch size for encoding.
        nlist: Number of Voronoi cells for IVF.
        m: Number of subquantizers for PQ.
        nbits: Bits per subquantizer.

    Returns:
        Tuple of (FAISS index, chunk mapping list).
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [f"passage: {chunk['text']}" for chunk in chunks]

    logger.info(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    logger.info(f"Building FAISS IVF-PQ index (dim={dim}, nlist={nlist}, m={m})...")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    train_size = min(len(embeddings), nlist * 40)
    train_indices = np.random.choice(len(embeddings), train_size, replace=False)
    index.train(embeddings[train_indices])

    index.add(embeddings)
    logger.info(f"Index built: {index.ntotal} vectors")

    chunk_mapping = [
        {"chunk_id": c["chunk_id"], "text": c["text"], **{k: v for k, v in c.items() if k not in ("chunk_id", "text")}}
        for c in chunks
    ]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_path))
        mapping_path = save_path.with_suffix(".mapping.pkl")
        with open(mapping_path, "wb") as f:
            pickle.dump(chunk_mapping, f)
        logger.info(f"Saved index to {save_path}")

    del model
    _free_memory()

    return index, chunk_mapping


def load_faiss_index(
    index_path: Path,
) -> tuple[Any, list[dict[str, Any]]]:
    """Load a FAISS index and chunk mapping from disk.

    Args:
        index_path: Path to the FAISS index file.

    Returns:
        Tuple of (FAISS index, chunk mapping list).
    """
    import faiss

    index_path = Path(index_path)
    index = faiss.read_index(str(index_path))
    mapping_path = index_path.with_suffix(".mapping.pkl")
    with open(mapping_path, "rb") as f:
        chunk_mapping = pickle.load(f)

    logger.info(f"Loaded index ({index.ntotal} vectors) from {index_path}")
    return index, chunk_mapping


def dense_search(
    query: str,
    index: Any,
    chunk_mapping: list[dict[str, Any]],
    model: Any,
    k: int = 50,
    nprobe: int = 16,
) -> list[RetrievalResult]:
    """Search the FAISS index with a query.

    Args:
        query: Query string.
        index: FAISS index.
        chunk_mapping: List of chunk metadata dicts.
        model: SentenceTransformer model instance.
        k: Number of results to return.
        nprobe: Number of cells to visit during search.

    Returns:
        List of RetrievalResult sorted by score descending.
    """
    index.nprobe = nprobe

    query_embedding = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunk_mapping[idx]
        results.append(RetrievalResult(
            chunk_id=chunk["chunk_id"],
            score=float(score),
            text=chunk["text"],
            metadata={k: v for k, v in chunk.items() if k not in ("chunk_id", "text")},
        ))

    return results


def load_embedding_model(model_name: str = "intfloat/multilingual-e5-large") -> Any:
    """Load a sentence-transformer model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        SentenceTransformer model instance.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _free_memory() -> None:
    """Free GPU memory."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
