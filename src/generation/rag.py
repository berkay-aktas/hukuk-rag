"""End-to-end RAG pipeline: question → retrieve → (rerank) → generate.

A single :class:`RagPipeline` holds all loaded components (FAISS index, BM25,
embedding model, optional reranker, LLM) and exposes one ``.answer(question)``
method that the benchmark runner and the demo both call.

Construct via :meth:`RagPipeline.from_paths` to load an index bundle written by
:mod:`src.pipeline.ingest`.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.generation.llm import LoadedLLM, generate_answer, load_llm
from src.generation.prompts import DEFAULT_SYSTEM, MCQ_SYSTEM, build_user_message
from src.reranker.cross_encoder import LoadedReranker, load_reranker, rerank
from src.retrieval.bm25 import bm25_search, load_bm25_index
from src.retrieval.dense import dense_search, load_embedding_model, load_faiss_index
from src.retrieval.fusion import rrf_merge
from src.retrieval.types import RetrievalResult

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class RagResponse:
    """Result of one query through the pipeline."""

    answer: str
    retrieved: list[RetrievalResult]
    reranked: list[RetrievalResult] | None = None
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class RagPipeline:
    """Loaded RAG components and the orchestration logic that combines them."""

    faiss_index: faiss.Index | None
    faiss_mapping: list[dict[str, Any]] | None
    bm25_index: Any | None
    bm25_mapping: list[dict[str, Any]] | None
    embed_model: SentenceTransformer | None
    llm: LoadedLLM
    reranker: LoadedReranker | None = None
    system_prompt: str = DEFAULT_SYSTEM

    # Retrieval params
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60
    rerank_top_k: int = 10
    final_top_k: int = 10

    # Generation params
    max_new_tokens: int = 256
    temperature: float = 0.1
    repetition_penalty: float = 1.2

    @classmethod
    def from_paths(
        cls,
        index_dir: Path | str,
        *,
        embedding_model: str | None = None,
        llm_base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        qlora_adapter: Path | str | None = None,
        reranker_model: Path | str | None = None,
        use_reranker: bool = False,
        use_bm25: bool = True,
        use_faiss: bool = True,
        system_prompt: str = DEFAULT_SYSTEM,
    ) -> RagPipeline:
        """Load a pipeline from an index bundle directory.

        Args:
            index_dir: A directory written by :func:`src.pipeline.ingest.build_indexes`.
            embedding_model: Override the embedding model. Defaults to the one
                recorded in the bundle manifest (or multilingual-e5-large).
            llm_base_model: HuggingFace model id for the base LLM.
            qlora_adapter: Optional path to a QLoRA adapter directory.
            reranker_model: HuggingFace id or local path. None = use off-shelf default.
            use_reranker: Whether to enable the reranker stage (slow + currently
                degrades quality if the FT adapter is the silver-label one).
            use_bm25: Include BM25 in the first-stage retrieval.
            use_faiss: Include FAISS in the first-stage retrieval.
            system_prompt: Which system prompt to use. Defaults to DEFAULT_SYSTEM.
        """
        import json

        index_dir = Path(index_dir)
        manifest_path = index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

        embed_model_id = embedding_model or manifest.get("embedding_model")

        # Dense
        faiss_index = faiss_mapping = embed_model = None
        if use_faiss:
            faiss_path = index_dir / "faiss.index"
            if not faiss_path.exists():
                raise FileNotFoundError(f"{faiss_path} missing — rebuild with build_indexes(build_faiss=True)")
            faiss_index, faiss_mapping = load_faiss_index(faiss_path)
            if embed_model_id is None:
                raise ValueError(
                    "Cannot infer embedding model. Pass embedding_model= explicitly "
                    "or ensure manifest.json records it."
                )
            embed_model = load_embedding_model(embed_model_id)

        # Sparse
        bm25_index = bm25_mapping = None
        if use_bm25:
            bm25_path = index_dir / "bm25.pkl"
            if not bm25_path.exists():
                raise FileNotFoundError(f"{bm25_path} missing — rebuild with build_indexes(build_bm25=True)")
            bm25_index, bm25_mapping = load_bm25_index(bm25_path)

        # LLM
        llm = load_llm(base_model=llm_base_model, adapter_path=qlora_adapter)

        # Reranker
        reranker = None
        if use_reranker:
            reranker = load_reranker(reranker_model if reranker_model else "seroe/bge-reranker-v2-m3-turkish-triplet")

        return cls(
            faiss_index=faiss_index,
            faiss_mapping=faiss_mapping,
            bm25_index=bm25_index,
            bm25_mapping=bm25_mapping,
            embed_model=embed_model,
            llm=llm,
            reranker=reranker,
            system_prompt=system_prompt,
        )

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """First-stage retrieval: dense + BM25 + RRF fusion."""
        result_lists: list[list[RetrievalResult]] = []

        if self.faiss_index is not None and self.embed_model is not None:
            dense_results = dense_search(
                query, self.faiss_index, self.faiss_mapping, self.embed_model, k=self.dense_top_k
            )
            result_lists.append(dense_results)

        if self.bm25_index is not None and self.bm25_mapping is not None:
            sparse_results = bm25_search(query, self.bm25_index, self.bm25_mapping, k=self.bm25_top_k)
            result_lists.append(sparse_results)

        if not result_lists:
            raise RuntimeError("No retrievers enabled — at least one of FAISS or BM25 must be loaded.")

        if len(result_lists) == 1:
            return result_lists[0]

        return rrf_merge(*result_lists, k=self.rrf_k)

    def answer(
        self,
        question: str,
        *,
        options: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> RagResponse:
        """Run the full pipeline on a single question.

        Args:
            question: The user question.
            options: Optional MCQ options. If provided, MCQ system prompt is used
                and the model is instructed to output ``Cevap: <letter>``.
            system_prompt: Override the configured system prompt for this call.
        """
        import time

        timing: dict[str, float] = {}

        t0 = time.time()
        retrieved = self.retrieve(question)
        timing["retrieve"] = round(time.time() - t0, 3)

        # Optional rerank
        reranked = None
        passages_for_llm = retrieved[: self.final_top_k]
        if self.reranker is not None and retrieved:
            t1 = time.time()
            # Send up to rerank_top_k from first-stage; truncate to final_top_k after
            candidates = retrieved[: max(self.rerank_top_k, self.final_top_k * 3)]
            reranked = rerank(self.reranker, question, candidates, top_k=self.final_top_k)
            timing["rerank"] = round(time.time() - t1, 3)
            passages_for_llm = reranked

        # Build prompt
        prompt_system = system_prompt or (MCQ_SYSTEM if options else self.system_prompt)
        user_msg = build_user_message(question, [r.text for r in passages_for_llm], options=options)

        # Generate
        t2 = time.time()
        answer_text = generate_answer(
            self.llm,
            system_prompt=prompt_system,
            user_message=user_msg,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
        )
        timing["generate"] = round(time.time() - t2, 3)

        return RagResponse(answer=answer_text, retrieved=retrieved, reranked=reranked, timing=timing)
