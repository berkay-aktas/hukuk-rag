"""Answer generation: LLM loading, prompt templates, and full RAG pipeline glue."""

from src.generation.llm import LoadedLLM, generate_answer, load_llm, smoke_test
from src.generation.prompts import (
    CITATION_STRICT_SYSTEM,
    DEFAULT_SYSTEM,
    MCQ_SYSTEM,
    SHORT_ANSWER_SYSTEM,
    build_user_message,
    format_context_block,
)
from src.generation.rag import RagPipeline, RagResponse

__all__ = [
    "CITATION_STRICT_SYSTEM",
    "DEFAULT_SYSTEM",
    "LoadedLLM",
    "MCQ_SYSTEM",
    "RagPipeline",
    "RagResponse",
    "SHORT_ANSWER_SYSTEM",
    "build_user_message",
    "format_context_block",
    "generate_answer",
    "load_llm",
    "smoke_test",
]
