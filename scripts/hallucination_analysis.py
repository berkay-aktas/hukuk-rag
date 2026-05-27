"""NLI-based hallucination analysis on a completed predictions.jsonl.

Implements the assignment's mandatory hallucination analysis. For every
sentence in each LLM-generated answer, compute its maximum NLI cosine
similarity against every sentence in the retrieved context. Sentences below
``--sim-threshold`` (default 0.50) are flagged as hallucinations — the model
asserted something the retrieval did not support.

Two complementary scores per question:

* **NLI faithfulness** — mean of the per-sentence max-similarity values.
  Continuous proxy for "how grounded is this answer in the retrieved
  context, semantically?" (vs token-overlap faithfulness which is purely
  lexical).
* **Hallucination rate** — fraction of LLM sentences below threshold.
  Discrete count of how often the model went off-source.

Aggregations reported (mandated by the assignment rubric):

* Corpus-level mean NLI faithfulness and hallucination rate.
* Per-domain breakdown (criminal, civil, commercial, administrative,
  constitutional) — uses ``metadata.domain`` if present.
* Per-difficulty breakdown (easy, medium, hard) — uses ``metadata.difficulty``.
* Correlation: question-level hallucination rate vs Token F1 (Spearman),
  to argue that the F1 we report is consistent with the faithfulness story.

Inputs / outputs mirror :mod:`scripts.nli_snap_counterfactual` — same
``--predictions``, ``--mapping``, ``--corpus``, ``--out`` flags so the same
Colab invocation pattern works.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import token_f1
from src.evaluation.nli import load_nli_scorer
from src.generation.postprocess import context_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunk_lookup(mapping_paths: list[Path], corpus_paths: list[Path]) -> dict[str, str]:
    """Same builder as the counterfactual script — duplicated to keep
    each script self-contained for Colab execution."""
    import pickle as _pickle

    lookup: dict[str, str] = {}
    for p in mapping_paths:
        with open(p, "rb") as f:
            chunks = _pickle.load(f)  # noqa: S301
        for c in chunks:
            cid = str(c.get("chunk_id"))
            txt = c.get("text") or ""
            if cid and cid not in lookup:
                lookup[cid] = txt
        logger.info("Loaded %d chunks from %s", len(chunks), p)

    for p in corpus_paths:
        added = 0
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
                    added += 1
        logger.info("Loaded %d chunks from %s", added, p)
    logger.info("Combined chunk lookup: %d unique ids", len(lookup))
    return lookup


def analyse(
    predictions: list[dict[str, Any]],
    chunk_lookup: dict[str, str],
    *,
    sim_threshold: float = 0.50,
    nli_model: str | None = None,
) -> dict[str, Any]:
    """Sentence-level NLI faithfulness + hallucination flagging."""
    scorer = load_nli_scorer(nli_model) if nli_model else load_nli_scorer()

    per_question: list[dict[str, Any]] = []
    started = time.time()

    for i, rec in enumerate(predictions):
        answer = rec.get("predicted_answer") or ""
        gold = rec.get("gold_answer") or ""
        ids = rec.get("retrieved_chunk_ids") or []
        ctx_texts = [chunk_lookup.get(str(cid), "") for cid in ids]
        ctx_texts = [t for t in ctx_texts if t]

        ans_sents = context_sentences([answer])
        ctx_sents = context_sentences(ctx_texts)

        if not ans_sents or not ctx_sents:
            # Edge: empty answer or empty context — record but don't crash.
            per_question.append({
                "question_id": rec.get("question_id"),
                "n_answer_sentences": len(ans_sents),
                "n_context_sentences": len(ctx_sents),
                "per_sentence_max_sim": [],
                "nli_faithfulness": 0.0,
                "hallucination_rate": 1.0 if ans_sents else 0.0,
                "token_f1": round(token_f1([answer], [gold]), 4),
                "metadata": rec.get("metadata"),
            })
            continue

        sim_matrix = scorer.pairwise_similarity(ans_sents, ctx_sents)
        per_sent_max = sim_matrix.max(axis=1)
        hallucinated_mask = per_sent_max < sim_threshold
        faith = float(per_sent_max.mean())
        hall_rate = float(hallucinated_mask.mean())

        per_question.append({
            "question_id": rec.get("question_id"),
            "n_answer_sentences": len(ans_sents),
            "n_context_sentences": len(ctx_sents),
            "per_sentence_max_sim": [round(float(s), 4) for s in per_sent_max],
            "per_sentence_text": ans_sents,
            "hallucinated_sentences": [
                ans_sents[j] for j in range(len(ans_sents)) if hallucinated_mask[j]
            ],
            "nli_faithfulness": round(faith, 4),
            "hallucination_rate": round(hall_rate, 4),
            "token_f1": round(token_f1([answer], [gold]), 4),
            "metadata": rec.get("metadata"),
        })

        if (i + 1) % 25 == 0:
            logger.info("  scored %d/%d in %.1fs", i + 1, len(predictions), time.time() - started)

    logger.info("Scored all %d in %.1fs", len(predictions), time.time() - started)

    # --- Aggregations -------------------------------------------------------
    faiths = np.asarray([q["nli_faithfulness"] for q in per_question])
    halls = np.asarray([q["hallucination_rate"] for q in per_question])
    f1s = np.asarray([q["token_f1"] for q in per_question])

    overall = {
        "n_questions": len(per_question),
        "nli_faithfulness_mean": round(float(faiths.mean()), 4),
        "nli_faithfulness_std": round(float(faiths.std()), 4),
        "hallucination_rate_mean": round(float(halls.mean()), 4),
        "hallucination_rate_std": round(float(halls.std()), 4),
        "sim_threshold": sim_threshold,
        "nli_model": scorer.model_id,
    }

    # Spearman correlation: hallucination rate vs Token F1.
    # We expect negative — more hallucination → lower F1.
    overall["corr_hallucination_vs_f1"] = round(_spearman(halls, f1s), 4)
    overall["corr_faithfulness_vs_f1"] = round(_spearman(faiths, f1s), 4)

    by_domain = _group_by(per_question, "domain")
    by_difficulty = _group_by(per_question, "difficulty")

    return {
        "overall": overall,
        "by_domain": by_domain,
        "by_difficulty": by_difficulty,
        "per_question": per_question,
    }


def _group_by(per_question: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Aggregate hallucination / faithfulness / F1 per metadata bucket."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for q in per_question:
        meta = q.get("metadata") or {}
        bucket = meta.get(key) or "unknown"
        buckets.setdefault(str(bucket), []).append(q)

    out: dict[str, dict[str, Any]] = {}
    for bucket, items in sorted(buckets.items()):
        faiths = np.asarray([q["nli_faithfulness"] for q in items])
        halls = np.asarray([q["hallucination_rate"] for q in items])
        f1s = np.asarray([q["token_f1"] for q in items])
        out[bucket] = {
            "n": len(items),
            "nli_faithfulness_mean": round(float(faiths.mean()), 4),
            "hallucination_rate_mean": round(float(halls.mean()), 4),
            "token_f1_mean": round(float(f1s.mean()), 4),
        }
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation — uses scipy if available, else a fallback."""
    if a.size != b.size or a.size < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(a, b)
        return float(0.0 if np.isnan(r) else r)
    except ImportError:
        # rank-based Pearson approximation
        ra = a.argsort().argsort().astype(float)
        rb = b.argsort().argsort().astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        denom = (ra.std() * rb.std()) or 1.0
        return float((ra * rb).mean() / denom)


def format_markdown(summary: dict[str, Any]) -> str:
    o = summary["overall"]
    lines = [
        "# NLI Hallucination Analysis",
        "",
        f"- **Questions:** {o['n_questions']}",
        f"- **NLI model:** `{o['nli_model']}`",
        f"- **Similarity threshold (hallucination = below):** {o['sim_threshold']:.2f}",
        f"- **Mean NLI faithfulness:** {o['nli_faithfulness_mean']:.4f} (± {o['nli_faithfulness_std']:.4f})",
        f"- **Mean hallucination rate:** {100*o['hallucination_rate_mean']:.2f}% (± {100*o['hallucination_rate_std']:.2f}pp)",
        f"- **Spearman ρ(hallucination, Token F1):** {o['corr_hallucination_vs_f1']:+.4f}",
        f"- **Spearman ρ(NLI faithfulness, Token F1):** {o['corr_faithfulness_vs_f1']:+.4f}",
        "",
        "## By domain",
        "",
        "| domain | n | NLI faithfulness | hallucination rate | Token F1 |",
        "|---|--:|---:|---:|---:|",
    ]
    for name, block in summary["by_domain"].items():
        lines.append(
            f"| {name} | {block['n']} | {block['nli_faithfulness_mean']:.4f} | "
            f"{100*block['hallucination_rate_mean']:.2f}% | {block['token_f1_mean']:.4f} |"
        )

    lines.extend(["", "## By difficulty", "",
                  "| difficulty | n | NLI faithfulness | hallucination rate | Token F1 |",
                  "|---|--:|---:|---:|---:|"])
    for name, block in summary["by_difficulty"].items():
        lines.append(
            f"| {name} | {block['n']} | {block['nli_faithfulness_mean']:.4f} | "
            f"{100*block['hallucination_rate_mean']:.2f}% | {block['token_f1_mean']:.4f} |"
        )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--mapping", type=Path, action="append", default=[])
    p.add_argument("--corpus", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sim-threshold", type=float, default=0.50,
                   help="NLI cosine similarity below this counts a sentence as hallucinated.")
    p.add_argument("--nli-model", type=str, default=None)
    args = p.parse_args(argv)

    if not args.mapping and not args.corpus:
        p.error("At least one of --mapping or --corpus is required.")

    predictions = load_predictions(args.predictions)
    logger.info("Loaded %d predictions from %s", len(predictions), args.predictions)

    chunk_lookup = load_chunk_lookup(args.mapping, args.corpus)
    if not chunk_lookup:
        raise RuntimeError("Empty chunk lookup — check --mapping/--corpus paths.")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = analyse(
        predictions,
        chunk_lookup,
        sim_threshold=args.sim_threshold,
        nli_model=args.nli_model,
    )

    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "summary.md").write_text(format_markdown(summary), encoding="utf-8")
    with open(args.out / "per_question.jsonl", "w", encoding="utf-8") as f:
        for rec in summary["per_question"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Wrote %s", args.out / "summary.json")
    logger.info("Wrote %s", args.out / "summary.md")
    logger.info("Wrote %s", args.out / "per_question.jsonl")


if __name__ == "__main__":
    main()
