"""End-to-end RAG pipeline: ingest corpora, run benchmarks, query interactively.

This package wraps the lower-level retrieval, generation, and evaluation modules
into surfaces designed for two consumers:

1. The course evaluator, who hands us their own document collection and
   benchmark Q&A file and expects scores back.
2. The Gradio demo, which needs a single object to call per query.
"""

from src.pipeline.ingest import build_indexes, detect_input_format

__all__ = ["build_indexes", "detect_input_format"]
