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
from src.generation.postprocess import snap_to_context_sentence
from src.generation.prompts import (
    CITATION_STRICT_SYSTEM,
    DEFAULT_SYSTEM,
    MCQ_SYSTEM,
    build_user_message,
)
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
    raw_answer: str | None = None        # pre-postprocess answer, when snap fired
    snap_proxy: float | None = None      # top-1 sentence overlap / pred-token-count


@dataclass
class RagPipeline:
    """Loaded RAG components and the orchestration logic that combines them.

    Supports retrieval over MULTIPLE corpora simultaneously: each corpus contributes
    its own FAISS dense index and/or BM25 sparse index, and results are RRF-merged
    across all sources. The embedding model is shared across all FAISS sources, so
    all dense indexes must have been built with the same model (or compatible ones).

    For a single corpus, pass single-item lists; for combined retrieval over Yargıtay
    + shared + Mevzuat, pass multiple sources.
    """

    # Dense + sparse sources. Each source is (faiss_index, mapping) or (bm25_index, mapping).
    dense_sources: list[tuple[Any, list[dict[str, Any]]]]
    bm25_sources: list[tuple[Any, list[dict[str, Any]]]]
    embed_model: SentenceTransformer | None
    llm: LoadedLLM
    reranker: LoadedReranker | None = None
    system_prompt: str = DEFAULT_SYSTEM

    # Per-source retrieval depth — used as top-k for each individual source before RRF
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60
    rerank_top_k: int = 30
    final_top_k: int = 10

    # Generation params. Defaults to greedy decoding (do_sample=False via temperature=0)
    # because sampling occasionally derails base Qwen 4-bit into CJK characters on noisy
    # Turkish legal contexts. Greedy is deterministic and stable; reproducibility wins
    # over diversity for benchmark eval.
    max_new_tokens: int = 256
    temperature: float = 0.0
    repetition_penalty: float = 1.2

    # Post-processor: snap-to-context-sentence. See src/generation/postprocess.py.
    # Empirically +0.015 F1 on top of off-shelf reranker at threshold 0.30.
    use_snap_postprocessor: bool = True
    snap_proxy_threshold: float = 0.30

    @classmethod
    def from_paths(
        cls,
        index_dir: Path | str | list[Path | str],
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
        """Load a pipeline from one or more index bundle directories.

        Args:
            index_dir: Either a single index bundle directory, or a list of them.
                When a list is provided, retrieval runs against ALL of them in
                parallel and results are RRF-merged.
            embedding_model: Override the embedding model. Required if any index
                bundle lacks a manifest.json.
            llm_base_model: HuggingFace model id for the base LLM.
            qlora_adapter: Optional path to a QLoRA adapter directory.
            reranker_model: HuggingFace id or local path. None = off-shelf default.
            use_reranker: Whether to enable the reranker stage.
            use_bm25: Include BM25 in the first-stage retrieval.
            use_faiss: Include FAISS in the first-stage retrieval.
            system_prompt: Which system prompt to use.
        """
        import json

        if isinstance(index_dir, (str, Path)):
            index_dirs = [Path(index_dir)]
        else:
            index_dirs = [Path(d) for d in index_dir]

        # Resolve embedding model: prefer explicit, then any manifest
        embed_model_id = embedding_model
        for d in index_dirs:
            manifest_path = d / "manifest.json"
            if manifest_path.exists() and embed_model_id is None:
                m = json.loads(manifest_path.read_text())
                embed_model_id = m.get("embedding_model")

        dense_sources: list[tuple[Any, list[dict[str, Any]]]] = []
        bm25_sources: list[tuple[Any, list[dict[str, Any]]]] = []
        embed_model = None

        if use_faiss:
            if embed_model_id is None:
                raise ValueError(
                    "Cannot infer embedding model. Pass embedding_model= explicitly "
                    "or ensure at least one index has manifest.json."
                )
            for d in index_dirs:
                faiss_path = d / "faiss.index"
                if faiss_path.exists():
                    dense_sources.append(load_faiss_index(faiss_path))
            if dense_sources:
                embed_model = load_embedding_model(embed_model_id)

        if use_bm25:
            for d in index_dirs:
                bm25_path = d / "bm25.pkl"
                if bm25_path.exists():
                    bm25_sources.append(load_bm25_index(bm25_path))

        if not dense_sources and not bm25_sources:
            raise FileNotFoundError(
                f"No faiss.index or bm25.pkl found in any of: {[str(d) for d in index_dirs]}"
            )

        llm = load_llm(base_model=llm_base_model, adapter_path=qlora_adapter)

        reranker = None
        if use_reranker:
            reranker = load_reranker(
                reranker_model if reranker_model else "seroe/bge-reranker-v2-m3-turkish-triplet"
            )

        return cls(
            dense_sources=dense_sources,
            bm25_sources=bm25_sources,
            embed_model=embed_model,
            llm=llm,
            reranker=reranker,
            system_prompt=system_prompt,
        )

    @classmethod
    def from_mixed_sources(
        cls,
        dense_index_paths: list[Path | str] = (),
        bm25_index_paths: list[Path | str] = (),
        *,
        embedding_model: str,
        llm_base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        qlora_adapter: Path | str | None = None,
        reranker_model: Path | str | None = None,
        use_reranker: bool = False,
        system_prompt: str = DEFAULT_SYSTEM,
    ) -> RagPipeline:
        """Load from EXPLICIT lists of faiss + bm25 paths.

        Use when sources live in different directories with different filename
        conventions (e.g. combining a manifest-aware bundle with a legacy
        Yargıtay layout where faiss is faiss_ft.index and bm25 is bm25.pkl
        in a sibling directory). Each path must have a matching .mapping.pkl
        sibling for the FAISS case.
        """
        dense_sources = [load_faiss_index(Path(p)) for p in dense_index_paths]
        bm25_sources = [load_bm25_index(Path(p)) for p in bm25_index_paths]

        embed_model = load_embedding_model(embedding_model) if dense_sources else None
        llm = load_llm(base_model=llm_base_model, adapter_path=qlora_adapter)
        reranker = None
        if use_reranker:
            reranker = load_reranker(
                reranker_model if reranker_model else "seroe/bge-reranker-v2-m3-turkish-triplet"
            )

        return cls(
            dense_sources=dense_sources,
            bm25_sources=bm25_sources,
            embed_model=embed_model,
            llm=llm,
            reranker=reranker,
            system_prompt=system_prompt,
        )

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """First-stage retrieval: query every source, RRF-merge across all."""
        result_lists: list[list[RetrievalResult]] = []

        if self.embed_model is not None:
            for faiss_idx, mapping in self.dense_sources:
                result_lists.append(
                    dense_search(query, faiss_idx, mapping, self.embed_model, k=self.dense_top_k)
                )

        for bm25_idx, mapping in self.bm25_sources:
            result_lists.append(bm25_search(query, bm25_idx, mapping, k=self.bm25_top_k))

        if not result_lists:
            raise RuntimeError("No retrievers enabled — load at least one FAISS or BM25 source.")

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

        # Snap post-processor — replace LLM paraphrase with the verbatim
        # context sentence it most overlaps with (when overlap is high).
        raw = None
        proxy = None
        if self.use_snap_postprocessor and passages_for_llm:
            t3 = time.time()
            snapped, proxy, fired = snap_to_context_sentence(
                answer_text,
                [r.text for r in passages_for_llm],
                proxy_threshold=self.snap_proxy_threshold,
            )
            if fired:
                raw = answer_text
                answer_text = snapped
            timing["postprocess"] = round(time.time() - t3, 3)

        return RagResponse(
            answer=answer_text,
            retrieved=retrieved,
            reranked=reranked,
            timing=timing,
            raw_answer=raw,
            snap_proxy=proxy,
        )
