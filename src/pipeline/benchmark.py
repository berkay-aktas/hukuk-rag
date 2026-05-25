"""Benchmark runner: run a Q&A file through a RAG pipeline, report metrics.

Accepts three benchmark schemas via auto-detection so the same CLI works on:

1. **Our gold set** — ``data/gold/gold_test_set.json``: top-level dict with
   ``"questions": [{question, gold_answer, madde_no, kaynak, ...}]``.
2. **Shared gold benchmark** — ``data/external/shared_2026/gold_benchmark.json``:
   array of ``{question_id, question, verified_answer, gold_sources: [{corpus_row_id, ...}]}``.
3. **Shared RAG eval** — ``data/external/shared_2026/rag_eval.json``: array of
   ``{query_id, query, gold_chunk_ids: [str], gold_answer_extract, ...}``.

Plus any JSONL file where each line has at least ``{question|query, answer|verified_answer}``.

When ``gold_chunk_ids`` (or ``gold_sources``) is present, retrieval metrics
(Recall@K, MRR, nDCG) are computed with real relevance labels. Otherwise they
fall back to None and only generation metrics are reported.

Output is a directory with:

::

    out/
    ├── predictions.jsonl    # per-question: question, gold, predicted, retrieved_ids, timing
    ├── summary.json         # aggregate metrics + bootstrap CIs
    └── summary.md           # human-readable markdown table
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.evaluation.metrics import (
    bootstrap_ci,
    citation_accuracy,
    faithfulness_score,
    generation_metrics,
    retrieval_metrics,
    token_f1,
)
from src.generation.rag import RagPipeline

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkExample:
    """One normalized benchmark question."""

    question_id: str
    question: str
    gold_answer: str
    options: list[str] | None = None
    gold_chunk_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None


def load_benchmark(path: Path | str) -> list[BenchmarkExample]:
    """Load a benchmark file into normalized :class:`BenchmarkExample` records.

    Auto-detects the three supported schemas (see module docstring).

    Args:
        path: Path to a ``.json`` or ``.jsonl`` benchmark file.

    Returns:
        A list of BenchmarkExample records, in file order.
    """
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "questions" in loaded:
            records = loaded["questions"]
        elif isinstance(loaded, list):
            records = loaded
        else:
            raise ValueError(f"Cannot parse benchmark schema at {p}: top-level is {type(loaded).__name__}")

    return [_normalize_example(r, fallback_id_prefix=p.stem, idx=i) for i, r in enumerate(records)]


def _normalize_example(rec: dict[str, Any], fallback_id_prefix: str, idx: int) -> BenchmarkExample:
    """Coerce one record from any supported schema into BenchmarkExample."""
    qid = rec.get("question_id") or rec.get("query_id") or rec.get("id") or f"{fallback_id_prefix}_{idx:05d}"
    question = rec.get("question") or rec.get("query")
    if not question:
        raise ValueError(f"Record {qid} has no question/query field")

    # Answer field aliases (in priority order)
    gold = (
        rec.get("gold_answer")
        or rec.get("verified_answer")
        or rec.get("gold_answer_extract")
        or rec.get("answer")
        or rec.get("expected_response")
        or ""
    )

    # Gold chunk ids: shared rag_eval uses gold_chunk_ids; gold_benchmark uses
    # gold_sources: [{corpus_row_id, source_id, ...}].
    gold_chunks: list[str] | None = None
    if "gold_chunk_ids" in rec and rec["gold_chunk_ids"]:
        gold_chunks = [str(x) for x in rec["gold_chunk_ids"]]
    elif "gold_sources" in rec and rec["gold_sources"]:
        gold_chunks = []
        for src in rec["gold_sources"]:
            if isinstance(src, dict):
                cid = src.get("corpus_row_id") or src.get("source_id") or src.get("chunk_id")
                if cid:
                    gold_chunks.append(str(cid))
            elif isinstance(src, str):
                gold_chunks.append(src)

    # Options for MCQ
    options = None
    if "options" in rec and isinstance(rec["options"], (list, dict)):
        if isinstance(rec["options"], list):
            options = [str(o) for o in rec["options"]]
        else:
            # Sometimes {A: "...", B: "..."}
            options = [rec["options"][k] for k in ("A", "B", "C", "D", "E") if k in rec["options"]]

    metadata = {k: v for k, v in rec.items() if k not in (
        "question_id", "query_id", "id", "question", "query",
        "gold_answer", "verified_answer", "gold_answer_extract", "answer", "expected_response",
        "gold_chunk_ids", "gold_sources", "options",
    )}

    return BenchmarkExample(
        question_id=str(qid),
        question=str(question),
        gold_answer=str(gold),
        options=options,
        gold_chunk_ids=gold_chunks,
        metadata=metadata or None,
    )


def run_benchmark(
    pipeline: RagPipeline,
    benchmark_path: Path | str,
    output_dir: Path | str,
    *,
    bootstrap_iterations: int = 1000,
    bootstrap_alpha: float = 0.05,
    save_predictions: bool = True,
) -> dict[str, Any]:
    """Run a benchmark through ``pipeline`` and write a metrics report.

    Args:
        pipeline: A :class:`RagPipeline` instance.
        benchmark_path: Path to a benchmark file (see module docstring for schemas).
        output_dir: Directory to write predictions + summary into. Created if missing.
        bootstrap_iterations: Number of bootstrap resamples for 95% CIs.
        bootstrap_alpha: Alpha for CI (default 0.05 → 95% CI).
        save_predictions: Write per-question predictions to predictions.jsonl.

    Returns:
        The summary dict (also written to ``output_dir/summary.json``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_benchmark(benchmark_path)
    logger.info("Loaded %d examples from %s", len(examples), benchmark_path)

    predictions: list[str] = []
    references: list[str] = []
    retrieved_ids: list[list[str]] = []
    relevant_ids: list[list[str]] = []
    contexts_joined: list[str] = []   # one string per prediction (passages joined)
    per_question_records: list[dict[str, Any]] = []

    # Resume support + crash safety: stream per-question records to disk as we go.
    # If the cell crashes mid-aggregation we still have everything on disk.
    preds_path = output_dir / "predictions.jsonl" if save_predictions else None
    if preds_path and preds_path.exists():
        preds_path.unlink()  # fresh start to avoid mixing runs

    started = time.time()
    for ex in tqdm(examples, desc="Benchmark"):
        try:
            resp = pipeline.answer(ex.question, options=ex.options)
        except Exception as e:
            logger.warning("Question %s failed: %s", ex.question_id, e)
            resp = None

        if resp is None:
            pred = ""
            retrieved = []
            reranked = []
            timing = {"error": True}
        else:
            pred = resp.answer
            retrieved = resp.retrieved
            reranked = resp.reranked or retrieved[: pipeline.final_top_k]
            timing = resp.timing

        predictions.append(pred)
        references.append(ex.gold_answer)
        retrieved_ids.append([r.chunk_id for r in retrieved])
        relevant_ids.append(ex.gold_chunk_ids or [])
        contexts_joined.append("\n\n".join(r.text for r in reranked))

        rec = {
            "question_id": ex.question_id,
            "question": ex.question,
            "gold_answer": ex.gold_answer,
            "predicted_answer": pred,
            "retrieved_chunk_ids": [r.chunk_id for r in reranked],
            "retrieved_scores": [round(r.score, 4) for r in reranked],
            "gold_chunk_ids": ex.gold_chunk_ids,
            "timing": timing,
            "metadata": ex.metadata,
        }
        per_question_records.append(rec)
        if preds_path:
            with open(preds_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_seconds = round(time.time() - started, 1)
    logger.info("Benchmark complete: %d questions in %.1fs", len(examples), total_seconds)

    # Aggregate metrics — wrap each in try/except so partial failure
    # still produces a usable summary instead of losing all the work.
    def _safe(label, fn):
        try:
            return fn()
        except Exception as e:
            logger.warning("Aggregate metric %s failed: %s", label, e)
            return None

    gen = _safe("generation", lambda: generation_metrics(predictions, references)) or {}
    faith_score = _safe("faithfulness", lambda: faithfulness_score(predictions, contexts_joined))
    faith = {"token_overlap": faith_score} if faith_score is not None else {}
    cite = _safe("citation", lambda: citation_accuracy(predictions, num_passages=pipeline.final_top_k)) or {}

    # Retrieval metrics — only if we have any chunk-level labels
    ret: dict[str, float] | None = None
    if any(relevant_ids):
        filtered_ret = [r for r, rel in zip(retrieved_ids, relevant_ids) if rel]
        filtered_rel = [rel for rel in relevant_ids if rel]
        ret = _safe("retrieval", lambda: retrieval_metrics(filtered_ret, filtered_rel))

    # Bootstrap CIs on Token F1
    f1_per_q = [token_f1([p], [r]) for p, r in zip(predictions, references)]
    point_f1 = sum(f1_per_q) / len(f1_per_q) if f1_per_q else 0.0
    boot = _safe(
        "bootstrap",
        lambda: bootstrap_ci(token_f1, predictions, references,
                             n=bootstrap_iterations, alpha=bootstrap_alpha),
    ) or {"mean": point_f1, "lower": point_f1, "upper": point_f1}

    summary = {
        "benchmark_path": str(benchmark_path),
        "n_questions": len(examples),
        "total_seconds": total_seconds,
        "mean_seconds_per_q": round(total_seconds / max(1, len(examples)), 2),
        "pipeline": {
            "embedding_model": getattr(pipeline.embed_model, "model_card_data", None) and pipeline.embed_model.model_card_data.base_model or "unknown",
            "llm_base": pipeline.llm.base_model_id,
            "qlora_adapter": pipeline.llm.adapter_path,
            "reranker": pipeline.reranker.model_id if pipeline.reranker else None,
            "system_prompt": pipeline.system_prompt[:120] + ("…" if len(pipeline.system_prompt) > 120 else ""),
        },
        "generation": gen,
        "faithfulness": faith,
        "citation": cite,
        "retrieval": ret,
        "token_f1_95ci": {"point": round(point_f1, 4), "ci_low": round(boot["lower"], 4), "ci_high": round(boot["upper"], 4)},
    }

    # Write outputs (predictions were already streamed during the loop)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    md_path = output_dir / "summary.md"
    md_path.write_text(_format_summary_markdown(summary), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    return summary


def _format_summary_markdown(summary: dict[str, Any]) -> str:
    """Render a summary dict as a tight Markdown report."""
    lines = [
        f"# Benchmark Summary",
        "",
        f"- **Benchmark:** `{summary['benchmark_path']}`",
        f"- **Questions:** {summary['n_questions']}",
        f"- **Total time:** {summary['total_seconds']}s ({summary['mean_seconds_per_q']}s/q)",
        f"- **LLM:** `{summary['pipeline']['llm_base']}`"
        + (f" + QLoRA adapter `{summary['pipeline']['qlora_adapter']}`" if summary['pipeline']['qlora_adapter'] else ""),
        f"- **Reranker:** {summary['pipeline']['reranker'] or 'none'}",
    ]
    def _emit_section(title: str, data: dict | None) -> None:
        if not data:
            return
        lines.extend(["", f"## {title}", "", "| Metric | Value |", "|---|---|"])
        for k, v in data.items():
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, (int, float)) else f"| {k} | {v} |")

    _emit_section("Generation", summary.get("generation"))
    ci = summary.get("token_f1_95ci") or {}
    if ci:
        lines.append(f"| token_f1 (bootstrap 95% CI) | {ci['point']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] |")
    _emit_section("Retrieval", summary.get("retrieval"))
    _emit_section("Faithfulness", summary.get("faithfulness"))
    _emit_section("Citation", summary.get("citation"))

    return "\n".join(lines) + "\n"
