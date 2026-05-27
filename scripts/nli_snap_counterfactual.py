"""Offline NLI-snap counterfactual on an existing predictions.jsonl.

Re-routes the snap post-processor on a completed benchmark run using the
NLI-gated router instead of the token-overlap proxy. Reports new Token F1
per threshold so we can pick the operating point that closes the most of
the 0.286 oracle ceiling without re-running the LLM.

Inputs:

* ``--predictions``: ``predictions.jsonl`` from the run we're counterfactual-ing.
  Each record must have ``predicted_answer``, ``gold_answer``, and
  ``retrieved_chunk_ids`` (the top-K passages that were sent to the LLM).
* ``--mapping`` (repeatable): pickle files mapping ``chunk_id -> {text, ...}``.
  These are the ``.mapping.pkl`` siblings written by the FAISS index build.
  Pass one per corpus so the union covers every chunk in retrieved_chunk_ids.
* ``--corpus`` (repeatable, optional): JSONL fallback. Each line is
  ``{id|chunk_id, text}``. Useful for the shared-2026 corpus where the
  pickle exposes the same data but a JSONL lookup is simpler.

The script does not invoke the LLM. It loads only the NLI scorer (~110M
params, runs on CPU in ~30s for 225 questions).

Output (``--out``):

* ``summary.json``: thresholds × F1 + bootstrap CIs.
* ``summary.md``: human-readable Markdown table.
* ``predictions.jsonl``: per-question snapped answers + chosen sentence.

Usage on Colab::

    !python -m scripts.nli_snap_counterfactual \\
        --predictions /content/drive/MyDrive/hukuk-rag/results/phase3_offshelf_reranker/predictions.jsonl \\
        --mapping /content/drive/MyDrive/hukuk-rag/indexes/phase0_shared_ft/faiss.mapping.pkl \\
        --mapping /content/drive/MyDrive/hukuk-rag/indexes/mevzuat_ft/faiss.mapping.pkl \\
        --out /content/drive/MyDrive/hukuk-rag/results/phase3_nli_snap_counterfactual
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as `python -m scripts.nli_snap_counterfactual` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import bootstrap_ci, token_f1
from src.evaluation.nli import load_nli_scorer
from src.generation.postprocess import snap_to_context_sentence_nli

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load a predictions.jsonl, one dict per question."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunk_lookup(mapping_paths: list[Path], corpus_paths: list[Path]) -> dict[str, str]:
    """Build a combined ``chunk_id -> text`` lookup across multiple sources.

    Mapping pickles store ``list[dict]`` where each dict has at minimum
    ``chunk_id`` and ``text``. JSONL corpora store one record per line with
    ``id`` (or ``chunk_id``) and ``text``. Later sources do not overwrite
    earlier ones — the first occurrence of an id wins.
    """
    lookup: dict[str, str] = {}

    for p in mapping_paths:
        with open(p, "rb") as f:
            chunks = pickle.load(f)  # noqa: S301
        added = 0
        for c in chunks:
            cid = str(c.get("chunk_id"))
            txt = c.get("text") or ""
            if cid and cid not in lookup:
                lookup[cid] = txt
                added += 1
        logger.info("Loaded %d/%d chunks from %s", added, len(chunks), p)

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


def run_counterfactual(
    predictions: list[dict[str, Any]],
    chunk_lookup: dict[str, str],
    *,
    thresholds: list[float],
    prefilter_top_k: int = 5,
    nli_model: str | None = None,
    bootstrap_n: int = 1000,
) -> dict[str, Any]:
    """Sweep ``thresholds`` and return per-threshold metrics + per-q snapped answers.

    Args:
        predictions: Records loaded from predictions.jsonl. Each must have
            ``predicted_answer``, ``gold_answer``, ``retrieved_chunk_ids``,
            and (optionally) ``metadata.domain`` / ``metadata.difficulty``.
        chunk_lookup: ``chunk_id -> passage text`` map.
        thresholds: NLI similarity thresholds to evaluate.
        prefilter_top_k: Top-K candidates by token overlap before NLI scoring.
        nli_model: Override the NLI sentence-transformer.
        bootstrap_n: Number of bootstrap resamples for the F1 CI per threshold.

    Returns:
        A dict with ``baseline``, ``per_threshold``, ``oracle``, and
        ``per_question`` for downstream inspection.
    """
    scorer = load_nli_scorer(nli_model) if nli_model else load_nli_scorer()

    base_preds = [r["predicted_answer"] for r in predictions]
    refs = [r["gold_answer"] for r in predictions]

    # Per-record retrieved texts (in reranked order, same as benchmark wrote)
    retrieved_texts: list[list[str]] = []
    missing_total = 0
    for rec in predictions:
        ids = rec.get("retrieved_chunk_ids") or []
        texts: list[str] = []
        for cid in ids:
            txt = chunk_lookup.get(str(cid))
            if txt:
                texts.append(txt)
            else:
                missing_total += 1
        retrieved_texts.append(texts)
    if missing_total:
        logger.warning(
            "Could not resolve %d chunk_ids across all questions — those passages "
            "are skipped. NLI router still runs on whatever resolved.",
            missing_total,
        )

    # Baseline F1 (no snap routing — just the LLM answer in predictions.jsonl).
    base_per_q = [token_f1([p], [r]) for p, r in zip(base_preds, refs)]
    base_mean = float(sum(base_per_q) / len(base_per_q)) if base_per_q else 0.0
    base_ci = bootstrap_ci(token_f1, base_preds, refs, n=bootstrap_n)

    logger.info("Baseline (no snap) Token F1: %.4f [%.4f, %.4f]",
                base_mean, base_ci["lower"], base_ci["upper"])

    # Per-threshold sweep
    per_threshold: dict[str, dict[str, Any]] = {}
    snapped_answers_by_threshold: dict[str, list[str]] = {}
    nli_sims_by_threshold: dict[str, list[float]] = {}

    # Compute NLI sim once per (question, candidate) tuple — sims don't depend
    # on threshold, only the routing decision does. We compute at the highest
    # operating threshold's candidate set and reuse.
    snap_signals: list[dict[str, Any]] = []
    started = time.time()
    for i, (ans, texts) in enumerate(zip(base_preds, retrieved_texts)):
        snapped, sim, fired = snap_to_context_sentence_nli(
            ans,
            texts,
            scorer,
            sim_threshold=0.0,  # threshold=0 → always returns best candidate
            prefilter_top_k=prefilter_top_k,
        )
        snap_signals.append({
            "snapped": snapped,
            "sim": float(sim),
            "fired_zero": fired,  # always True when there's anything to score
            "base": ans,
        })
        if (i + 1) % 50 == 0:
            logger.info("  scored %d/%d in %.1fs", i + 1, len(base_preds), time.time() - started)
    logger.info("Scored all %d in %.1fs", len(base_preds), time.time() - started)

    # Now apply each threshold
    for thr in thresholds:
        new_preds = []
        per_q_sims = []
        snap_count = 0
        for sig in snap_signals:
            per_q_sims.append(sig["sim"])
            if sig["sim"] >= thr:
                new_preds.append(sig["snapped"])
                snap_count += 1
            else:
                new_preds.append(sig["base"])

        per_q = [token_f1([p], [r]) for p, r in zip(new_preds, refs)]
        mean = float(sum(per_q) / len(per_q)) if per_q else 0.0
        ci = bootstrap_ci(token_f1, new_preds, refs, n=bootstrap_n)

        per_threshold[f"{thr:.2f}"] = {
            "threshold": thr,
            "mean_f1": round(mean, 4),
            "f1_95ci_low": round(ci["lower"], 4),
            "f1_95ci_high": round(ci["upper"], 4),
            "delta_vs_baseline": round(mean - base_mean, 4),
            "snapped_count": snap_count,
            "snapped_fraction": round(snap_count / len(predictions), 4),
        }
        snapped_answers_by_threshold[f"{thr:.2f}"] = new_preds
        nli_sims_by_threshold[f"{thr:.2f}"] = per_q_sims
        logger.info("thr=%.2f  F1=%.4f [%.4f, %.4f]  snapped=%d/%d  Δ=%+.4f",
                    thr, mean, ci["lower"], ci["upper"], snap_count, len(predictions),
                    mean - base_mean)

    # Oracle: per-question max(base, snapped) — assumes a perfect router.
    oracle_preds = [
        sig["snapped"] if token_f1([sig["snapped"]], [refs[i]]) > token_f1([sig["base"]], [refs[i]])
        else sig["base"]
        for i, sig in enumerate(snap_signals)
    ]
    oracle_per_q = [token_f1([p], [r]) for p, r in zip(oracle_preds, refs)]
    oracle_mean = float(sum(oracle_per_q) / len(oracle_per_q))
    oracle_ci = bootstrap_ci(token_f1, oracle_preds, refs, n=bootstrap_n)
    logger.info("ORACLE (perfect NLI router) F1=%.4f [%.4f, %.4f] Δ=%+.4f",
                oracle_mean, oracle_ci["lower"], oracle_ci["upper"], oracle_mean - base_mean)

    # Pick best threshold by mean F1
    best_thr_key = max(per_threshold.keys(), key=lambda k: per_threshold[k]["mean_f1"])
    best_block = per_threshold[best_thr_key]
    logger.info("BEST threshold=%.2f → F1=%.4f", best_block["threshold"], best_block["mean_f1"])

    # Per-question records (best threshold only — predictions.jsonl-shaped)
    best_preds = snapped_answers_by_threshold[best_thr_key]
    best_sims = nli_sims_by_threshold[best_thr_key]
    per_question = []
    for i, rec in enumerate(predictions):
        per_question.append({
            "question_id": rec.get("question_id"),
            "gold_answer": rec.get("gold_answer"),
            "base_predicted": rec.get("predicted_answer"),
            "snapped_predicted": best_preds[i],
            "nli_sim": round(best_sims[i], 4),
            "was_snapped": bool(best_preds[i] != rec.get("predicted_answer")),
            "base_f1": round(base_per_q[i], 4),
            "snapped_f1": round(token_f1([best_preds[i]], [refs[i]]), 4),
            "oracle_f1": round(oracle_per_q[i], 4),
            "metadata": rec.get("metadata"),
        })

    return {
        "baseline": {
            "mean_f1": round(base_mean, 4),
            "f1_95ci_low": round(base_ci["lower"], 4),
            "f1_95ci_high": round(base_ci["upper"], 4),
        },
        "per_threshold": per_threshold,
        "best_threshold_key": best_thr_key,
        "best_threshold": best_block,
        "oracle": {
            "mean_f1": round(oracle_mean, 4),
            "f1_95ci_low": round(oracle_ci["lower"], 4),
            "f1_95ci_high": round(oracle_ci["upper"], 4),
            "delta_vs_baseline": round(oracle_mean - base_mean, 4),
        },
        "n_questions": len(predictions),
        "thresholds_swept": list(thresholds),
        "prefilter_top_k": prefilter_top_k,
        "nli_model": scorer.model_id,
        "per_question": per_question,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# NLI Snap Router Counterfactual",
        "",
        f"- **Questions:** {summary['n_questions']}",
        f"- **NLI model:** `{summary['nli_model']}`",
        f"- **Prefilter top-K (by token overlap):** {summary['prefilter_top_k']}",
        f"- **Baseline Token F1 (no snap):** {summary['baseline']['mean_f1']:.4f} "
        f"[{summary['baseline']['f1_95ci_low']:.4f}, {summary['baseline']['f1_95ci_high']:.4f}]",
        f"- **Oracle (perfect NLI router):** {summary['oracle']['mean_f1']:.4f} "
        f"[{summary['oracle']['f1_95ci_low']:.4f}, {summary['oracle']['f1_95ci_high']:.4f}] "
        f"(Δ = {summary['oracle']['delta_vs_baseline']:+.4f})",
        f"- **Best operating threshold:** {summary['best_threshold']['threshold']:.2f}",
        "",
        "## Threshold sweep",
        "",
        "| threshold | mean F1 | 95% CI | Δ vs baseline | snapped |",
        "|---:|---:|---|---:|---:|",
    ]
    for key, block in summary["per_threshold"].items():
        lines.append(
            f"| {block['threshold']:.2f} | {block['mean_f1']:.4f} | "
            f"[{block['f1_95ci_low']:.4f}, {block['f1_95ci_high']:.4f}] | "
            f"{block['delta_vs_baseline']:+.4f} | "
            f"{block['snapped_count']}/{summary['n_questions']} "
            f"({100*block['snapped_fraction']:.0f}%) |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--predictions", type=Path, required=True,
                   help="predictions.jsonl from the run being counterfactual-ed.")
    p.add_argument("--mapping", type=Path, action="append", default=[],
                   help="Pickle mapping file (chunk_id -> text). Repeatable.")
    p.add_argument("--corpus", type=Path, action="append", default=[],
                   help="JSONL corpus fallback. Repeatable.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory; created if missing.")
    p.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        help="NLI similarity thresholds to evaluate.",
    )
    p.add_argument("--prefilter-top-k", type=int, default=5,
                   help="Token-overlap pre-filter shortlist size before NLI.")
    p.add_argument("--nli-model", type=str, default=None,
                   help="Override the NLI sentence-transformer id.")
    p.add_argument("--bootstrap-n", type=int, default=1000,
                   help="Bootstrap resamples for the F1 CI.")
    args = p.parse_args(argv)

    if not args.mapping and not args.corpus:
        p.error("At least one of --mapping or --corpus is required.")

    predictions = load_predictions(args.predictions)
    logger.info("Loaded %d predictions from %s", len(predictions), args.predictions)

    chunk_lookup = load_chunk_lookup(args.mapping, args.corpus)
    if not chunk_lookup:
        raise RuntimeError("Empty chunk lookup — check --mapping/--corpus paths.")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = run_counterfactual(
        predictions,
        chunk_lookup,
        thresholds=sorted(args.thresholds),
        prefilter_top_k=args.prefilter_top_k,
        nli_model=args.nli_model,
        bootstrap_n=args.bootstrap_n,
    )

    # Write outputs
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    md_path = args.out / "summary.md"
    md_path.write_text(format_markdown(summary), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    per_q_path = args.out / "predictions.jsonl"
    with open(per_q_path, "w", encoding="utf-8") as f:
        for rec in summary["per_question"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d records)", per_q_path, len(summary["per_question"]))


if __name__ == "__main__":
    main()
