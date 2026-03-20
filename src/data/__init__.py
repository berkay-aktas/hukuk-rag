"""Data ingestion, chunking, and gold-set management."""

from src.data.chunker import Chunk, chunk_corpus, chunk_parquet_streaming, legal_aware_chunk
from src.data.gold_set import gold_set_stats, load_gold_set, validate_gold_set
from src.data.preprocessor import load_corpus, save_corpus

__all__ = [
    "Chunk",
    "chunk_corpus",
    "chunk_parquet_streaming",
    "legal_aware_chunk",
    "gold_set_stats",
    "load_gold_set",
    "validate_gold_set",
    "load_corpus",
    "save_corpus",
]
