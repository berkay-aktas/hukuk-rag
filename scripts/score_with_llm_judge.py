"""Score a predictions.jsonl with the LLM-as-judge and write composite metrics.

The script ties together :mod:`src.evaluation.llm_judge` and the chunk-lookup
plumbing from the NLI counterfactual / hallucination analysis scripts:

1. Load a benchmark's ``predictions.jsonl``.
2. Resolve retrieved chunk texts via one or more mapping pickles (so the
   judge can see the same context the LLM saw).
3. For each record, call the judge LLM once and parse JSON scores.
4. Aggregate axes overall + per (domain, difficulty); write summary.json,
   summary.md, and per_question_judge.jsonl.
5. Optionally compute the rubric's composite scenario scores when recall@k
   / semantic similarity values are passed on the command line.

The judge LLM defaults to ``Qwen/Qwen2.5-7B-Instruct``. Pass
``--judge-model`` to use a different one; pass ``--judge-adapter`` to use a
PEFT adapter on top of the base.

Usage on Colab::

    !python -m scripts.score_with_llm_judge \\
        --predictions /content/drive/MyDrive/hukuk-rag/results/phase3_offshelf_reranker/predictions.jsonl \\
        --mapping /content/drive/MyDrive/hukuk-rag/indexes/phase0_shared_ft/faiss.mapping.pkl \\
        --mapping /content/drive/MyDrive/hukuk-rag/indexes/mevzuat_ft/faiss.mapping.pkl \\
        --mapping /content/drive/MyDrive/hukuk-rag/indexes/faiss.mapping.pkl \\
        --out /content/drive/MyDrive/hukuk-rag/results/phase3_llm_judge \\
        --semantic-similarity 0.7026
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.llm_judge import (
    aggregate_scores,
    composite_score,
    score_predictions,
)
from src.generation.llm import load_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunk_lookup(mapping_paths: list[Path], corpus_paths: list[Path]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for p in mapping_paths:
        with open(p, "rb") as f:
            chunks = pickle.load(f)  # noqa: S301
        for c in chunks:
            cid = str(c.get("chunk_id"))
            txt = c.get("text") or ""
            if cid and cid not in lookup:
                lookup[cid] = txt
        logger.info("Loaded %d chunks from %s", len(chunks), p)
    for p in corpus_paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = str(rec.get("id") or rec.get("chunk_id") or "")
                txt = rec.get("text") or ""
                if cid and cid not in lookup:
                    lookup[cid] = txt
        logger.info("Loaded chunks from %s", p)
    logger.info("Combined chunk lookup: %d unique ids", len(lookup))
    return lookup


def format_markdown(summary: dict[str, Any]) -> str:
    h = summary["aggregate"]["overall"]
    lines = [
        "# LLM-as-Judge scores",
        "",
        f"- **Predictions:** `{summary['predictions_path']}`",
        f"- **Judge model:** `{summary['judge_model']}`",
        f"- **Questions scored:** {summary['n_questions']}",
        "",
        "## Per-axis means",
        "",
        "| axis | mean | scored |",
        "|---|---:|---:|",
    ]
    for axis in ("correctness", "faithfulness", "coherence", "relevancy"):
        block = h.get(axis) or {}
        mean = block.get("mean")
        if mean is None:
            lines.append(f"| {axis} | n/a | {block.get('n_scored', 0)}/{block.get('n_total', 0)} |")
        else:
            lines.append(f"| {axis} | {mean:.4f} | {block['n_scored']}/{block['n_total']} |")

    if summary.get("composite"):
        lines += ["", "## Rubric composite scores", "", "| scenario | score | notes |", "|---|---:|---|"]
        c = summary["composite"]
        if c.get("scenario_1") is not None:
            lines.append(f"| 1 (R+A+G) | {c['scenario_1']:.4f} | 0.35*R + 0.40*A + 0.25*G |")
        if c.get("scenario_2") is not None:
            lines.append(f"| 2 (A+Sim) | {c['scenario_2']:.4f} | 0.70*A + 0.30*Sim |")
        if c.get("scenario_3") is not None:
            lines.append(f"| 3 (LLM avg) | {c['scenario_3']:.4f} | avg(faithfulness, coherence, relevancy) |")

    grp = summary["aggregate"].get("by_group")
    if grp:
        lines += ["", "## By group", "", "| group | n | correctness | faithfulness | coherence | relevancy |",
                  "|---|--:|---:|---:|---:|---:|"]
        for name, b in grp.items():
            def _f(v):
                return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
            lines.append(
                f"| {name} | {b['n']} | {_f(b.get('correctness'))} | "
                f"{_f(b.get('faithfulness'))} | {_f(b.get('coherence'))} | "
                f"{_f(b.get('relevancy'))} |"
            )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--mapping", type=Path, action="append", default=[])
    p.add_argument("--corpus", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--judge-adapter", type=Path, default=None,
                   help="Optional PEFT adapter for the judge model.")
    p.add_argument("--recall-at-k", type=float, default=None,
                   help="Pass-through value to compute Scenario 1 composite.")
    p.add_argument("--semantic-similarity", type=float, default=None,
                   help="Pass-through (e.g., BERTScore F1) for Scenario 2 composite.")
    args = p.parse_args(argv)

    records = load_predictions(args.predictions)
    logger.info("Loaded %d predictions from %s", len(records), args.predictions)

    chunk_lookup = load_chunk_lookup(args.mapping, args.corpus) if (args.mapping or args.corpus) else {}

    judge_llm = load_llm(base_model=args.judge_model, adapter_path=args.judge_adapter)

    scores = score_predictions(judge_llm, records, chunk_lookup=chunk_lookup)
    metadata = [r.get("metadata") for r in records]
    aggregate = aggregate_scores(scores, grouping_metadata=metadata)

    composite = composite_score(
        aggregate,
        recall_at_k=args.recall_at_k,
        semantic_similarity=args.semantic_similarity,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "predictions_path": str(args.predictions),
        "judge_model": args.judge_model,
        "judge_adapter": str(args.judge_adapter) if args.judge_adapter else None,
        "n_questions": len(records),
        "aggregate": aggregate,
        "composite": composite,
        "inputs": {
            "recall_at_k": args.recall_at_k,
            "semantic_similarity": args.semantic_similarity,
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "summary.md").write_text(format_markdown(summary), encoding="utf-8")
    with open(args.out / "per_question_judge.jsonl", "w", encoding="utf-8") as f:
        for rec, s in zip(records, scores):
            f.write(json.dumps({
                "question_id": rec.get("question_id"),
                "scores": asdict(s),
            }, ensure_ascii=False) + "\n")

    logger.info("Wrote %s", args.out / "summary.json")
    logger.info("Wrote %s", args.out / "summary.md")
    logger.info("Wrote %s", args.out / "per_question_judge.jsonl")
    logger.info("Headline: %s", json.dumps(aggregate["overall"], indent=2))


if __name__ == "__main__":
    main()
