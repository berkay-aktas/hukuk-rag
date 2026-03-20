"""Retrieval: dense (FAISS), sparse (BM25), and hybrid fusion."""

from src.retrieval.types import RetrievalResult
from src.retrieval.dense import build_faiss_index, dense_search, load_embedding_model, load_faiss_index
from src.retrieval.bm25 import bm25_search, build_bm25_index, load_bm25_index
from src.retrieval.fusion import rrf_merge

__all__ = [
    "RetrievalResult",
    "build_faiss_index",
    "dense_search",
    "load_embedding_model",
    "load_faiss_index",
    "bm25_search",
    "build_bm25_index",
    "load_bm25_index",
    "rrf_merge",
]
