"""Offline citation-snap counterfactual on an existing predictions.jsonl.

Sweeps the citation-extraction router against the proxy/NLI sentence routers
on a completed benchmark run, so we can pick the best combination rule
without re-running the LLM.

The script loads ``predictions.jsonl``, resolves each ``retrieved_chunk_ids``
list to chunk texts via one or more mapping pickles, then for each question
evaluates these rules:

* ``baseline`` — base LLM answer as-is (no snap).
* ``sentence_only`` — the existing proxy-only token-overlap router.
* ``citation_only`` — citation-snap only; fires when an LLM citation is
  grounded in retrieval, otherwise keeps the base answer.
* ``citation_first`` — citation-snap if grounded; else fall through to the
  sentence router.
* ``sentence_first`` — sentence router if it fires; else fall through to
  citation-snap.
* ``oracle`` — per-question max F1 across {base, sentence, citation}.

Reports per-rule Token F1 with bootstrap 95% CIs, plus per-question
diagnostics (which rules fired, which citations were extracted, what was
snapped to).

The script does not load any LLM. It runs entirely on CPU in ~1-2 min for
225 questions.

Usage on Colab::

    !python -m scripts.citation_snap_counterfactual \\
        --predictions /content/drive/MyDrive/hukuk-rag/results/phase6_qwen14b_drop_in/predictions.jsonl \\
        --mapping /content/drive/MyDrive/hukuk-rag/indexes/phase0_shared_ft/faiss.mapping.pkl \\
        --mapping /content/drive/MyDrive/hukuk-rag/indexes/mevzuat_ft/faiss.mapping.pkl \\
        --mapping /content/drive/MyDrive/hukuk-rag/indexes/faiss.mapping.pkl \\
        --out /content/drive/MyDrive/hukuk-rag/results/phase6_citation_snap_counterfactual

If the predictions came from a pipeline with ``snap_skip_sources={'yargitay'}``
(the production default), pass ``--skip-source yargitay`` so the counterfactual
honours the same filter. Sources are derived from each chunk's metadata via
:func:`src.retrieval.types.infer_source`.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import bootstrap_ci, token_f1
from src.generation.postprocess import (
    extract_citations,
    snap_to_cited_madde,
    snap_to_context_sentence,
)
from src.retrieval.types import RetrievalResult, infer_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load a predictions.jsonl, one dict per question."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunk_lookup(
    mapping_paths: list[Path],
    corpus_paths: list[Path],
) -> dict[str, dict[str, Any]]:
    """Build ``chunk_id -> {text, metadata}`` across mapping pickles + JSONL corpora.

    Mapping pickles store ``list[dict]`` where each dict has at minimum
    ``chunk_id`` and ``text``; other keys (source, source_file, _source) are
    preserved as metadata so we can later filter by source. First occurrence
    of an id wins, so order mapping paths by trust/freshness.
    """
    lookup: dict[str, dict[str, Any]] = {}

    for p in mapping_paths:
        with open(p, "rb") as f:
            chunks = pickle.load(f)  # noqa: S301
        added = 0
        for c in chunks:
            cid = str(c.get("chunk_id"))
            txt = c.get("text") or ""
            if cid and cid not in lookup:
                lookup[cid] = {
                    "text": txt,
                    # Preserve heterogeneous metadata keys for infer_source.
                    "metadata": {k: v for k, v in c.items() if k not in ("chunk_id", "text")},
                }
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
                    lookup[cid] = {
                        "text": txt,
                        "metadata": {k: v for k, v in rec.items() if k not in ("id", "chunk_id", "text")},
                    }
                    added += 1
        logger.info("Loaded %d chunks from %s", added, p)

    logger.info("Combined chunk lookup: %d unique ids", len(lookup))
    return lookup


def resolve_retrieved_texts(
    rec: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
    skip_sources: set[str],
) -> list[str]:
    """Resolve a record's ``retrieved_chunk_ids`` to passage texts.

    Applies ``skip_sources`` filter mirroring ``RagPipeline.snap_skip_sources``
    so the counterfactual sees exactly the same candidate pool the live
    pipeline would have. Unknown sources fall through (treated as not-skipped).
    """
    out: list[str] = []
    for cid in rec.get("retrieved_chunk_ids") or []:
        entry = chunk_lookup.get(str(cid))
        if not entry:
            continue
        # Synthesize a RetrievalResult so we can reuse infer_source.
        rr = RetrievalResult(
            chunk_id=str(cid),
            score=0.0,
            text=entry["text"],
            metadata=entry["metadata"],
        )
        if skip_sources and infer_source(rr) in skip_sources:
            continue
        if entry["text"]:
            out.append(entry["text"])
    return out


def evaluate_rules(
    predictions: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    *,
    skip_sources: set[str],
    proxy_threshold: float,
    bootstrap_n: int,
) -> dict[str, Any]:
    """Evaluate every rule on every question and return aggregated metrics.

    Computes per-rule predictions in a single pass over the questions, then
    bootstraps F1 CIs at the end. The rule logic mirrors
    :func:`src.generation.postprocess.snap_route_decision` exactly so the
    counterfactual numbers are directly applicable to the live pipeline.
    """
    refs = [r["gold_answer"] for r in predictions]
    base_preds = [r["predicted_answer"] for r in predictions]

    # Per-rule predictions (parallel arrays, same length as predictions)
    rule_preds: dict[str, list[str]] = {
        "baseline": list(base_preds),
        "sentence_only": [],
        "citation_only": [],
        "citation_first": [],
        "sentence_first": [],
        "oracle": [],
    }

    # Per-question diagnostics for the report
    per_q: list[dict[str, Any]] = []
    fire_counts: Counter[str] = Counter()
    missing_chunks_total = 0
    citation_rate = 0  # questions with any extractable citation

    started = time.time()
    for i, rec in enumerate(predictions):
        base = base_preds[i]
        texts = resolve_retrieved_texts(rec, chunk_lookup, skip_sources)
        missing = (len(rec.get("retrieved_chunk_ids") or []) - len(texts))
        missing_chunks_total += missing

        # ---- Citation-snap signals ----
        citations = extract_citations(base)
        if citations:
            citation_rate += 1
        cit_ans, _, cit_fired = snap_to_cited_madde(base, texts)
        if cit_fired:
            fire_counts["citation"] += 1

        # ---- Sentence-snap signals (proxy-only, fixed threshold) ----
        sent_ans, sent_proxy, sent_fired = snap_to_context_sentence(
            base, texts, proxy_threshold=proxy_threshold,
        )
        if sent_fired:
            fire_counts["sentence"] += 1

        # ---- Compose rule predictions ----
        rule_preds["sentence_only"].append(sent_ans if sent_fired else base)
        rule_preds["citation_only"].append(cit_ans if cit_fired else base)
        if cit_fired:
            rule_preds["citation_first"].append(cit_ans)
        elif sent_fired:
            rule_preds["citation_first"].append(sent_ans)
        else:
            rule_preds["citation_first"].append(base)
        if sent_fired:
            rule_preds["sentence_first"].append(sent_ans)
        elif cit_fired:
            rule_preds["sentence_first"].append(cit_ans)
        else:
            rule_preds["sentence_first"].append(base)

        # Oracle: per-q max F1 across {base, sentence, citation}.
        oracle_candidates = [(base, "base")]
        if sent_fired:
            oracle_candidates.append((sent_ans, "sentence"))
        if cit_fired:
            oracle_candidates.append((cit_ans, "citation"))
        oracle_ans, oracle_kind = max(
            oracle_candidates,
            key=lambda pair: token_f1([pair[0]], [refs[i]]),
        )
        rule_preds["oracle"].append(oracle_ans)

        per_q.append({
            "question_id": rec.get("question_id"),
            "gold": rec.get("gold_answer"),
            "base": base,
            "metadata": rec.get("metadata"),
            "n_retrieved_after_skip": len(texts),
            "n_missing_chunks": missing,
            "citations": citations,
            "citation_fired": cit_fired,
            "citation_snap": cit_ans if cit_fired else None,
            "sentence_fired": sent_fired,
            "sentence_proxy": round(float(sent_proxy), 4),
            "sentence_snap": sent_ans if sent_fired else None,
            "oracle_kind": oracle_kind,
            "base_f1": round(token_f1([base], [refs[i]]), 4),
            "citation_only_f1": round(token_f1([cit_ans if cit_fired else base], [refs[i]]), 4),
            "sentence_only_f1": round(token_f1([sent_ans if sent_fired else base], [refs[i]]), 4),
            "citation_first_f1": round(token_f1([rule_preds["citation_first"][-1]], [refs[i]]), 4),
            "sentence_first_f1": round(token_f1([rule_preds["sentence_first"][-1]], [refs[i]]), 4),
            "oracle_f1": round(token_f1([oracle_ans], [refs[i]]), 4),
        })

        if (i + 1) % 50 == 0:
            logger.info("  scored %d/%d in %.1fs", i + 1, len(predictions), time.time() - started)

    logger.info("Scored all %d in %.1fs", len(predictions), time.time() - started)
    if missing_chunks_total:
        logger.warning(
            "Could not resolve %d chunk_ids across all questions — those passages "
            "were treated as empty in this counterfactual.",
            missing_chunks_total,
        )

    # Aggregate per-rule
    per_rule: dict[str, dict[str, Any]] = {}
    base_mean = float(sum(token_f1([p], [r]) for p, r in zip(base_preds, refs)) / max(1, len(refs)))
    for name, preds in rule_preds.items():
        mean = float(sum(token_f1([p], [r]) for p, r in zip(preds, refs)) / max(1, len(refs)))
        ci = bootstrap_ci(token_f1, preds, refs, n=bootstrap_n)
        per_rule[name] = {
            "mean_f1": round(mean, 4),
            "f1_95ci_low": round(ci["lower"], 4),
            "f1_95ci_high": round(ci["upper"], 4),
            "delta_vs_baseline": round(mean - base_mean, 4),
        }
        logger.info(
            "rule=%-16s  F1=%.4f [%.4f, %.4f]  Δ=%+.4f",
            name, mean, ci["lower"], ci["upper"], mean - base_mean,
        )

    # Best rule (excluding oracle, which isn't shippable)
    shippable = {k: v for k, v in per_rule.items() if k != "oracle"}
    best_name = max(shippable.keys(), key=lambda k: shippable[k]["mean_f1"])
    logger.info(
        "BEST shippable rule: %s (F1=%.4f, Δ=%+.4f)",
        best_name, per_rule[best_name]["mean_f1"], per_rule[best_name]["delta_vs_baseline"],
    )

    return {
        "n_questions": len(predictions),
        "proxy_threshold": proxy_threshold,
        "skip_sources": sorted(skip_sources),
        "missing_chunks_total": missing_chunks_total,
        "citation_extraction_rate": round(citation_rate / max(1, len(predictions)), 4),
        "fire_counts": dict(fire_counts),
        "per_rule": per_rule,
        "best_shippable_rule": best_name,
        "per_question": per_q,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    rules_block = summary["per_rule"]
    base = rules_block["baseline"]
    lines = [
        "# Citation-Snap Counterfactual",
        "",
        f"- **Questions:** {summary['n_questions']}",
        f"- **Proxy threshold (sentence-snap):** {summary['proxy_threshold']:.2f}",
        f"- **Snap source filter:** {summary['skip_sources'] or '(none)'}",
        f"- **Citation extraction rate (any citation in answer):** "
        f"{100*summary['citation_extraction_rate']:.1f}%",
        f"- **Missing chunk_ids (treated as empty):** "
        f"{summary['missing_chunks_total']}",
        "",
        "## Fire counts",
        "",
        "| router | fired |",
        "|---|---:|",
        f"| citation | {summary['fire_counts'].get('citation', 0)}/{summary['n_questions']} |",
        f"| sentence | {summary['fire_counts'].get('sentence', 0)}/{summary['n_questions']} |",
        "",
        "## Per-rule Token F1",
        "",
        "| rule | mean F1 | 95% CI | Δ vs baseline |",
        "|---|---:|---|---:|",
    ]
    for name in ("baseline", "sentence_only", "citation_only",
                 "citation_first", "sentence_first", "oracle"):
        b = rules_block[name]
        lines.append(
            f"| `{name}` | {b['mean_f1']:.4f} | "
            f"[{b['f1_95ci_low']:.4f}, {b['f1_95ci_high']:.4f}] | "
            f"{b['delta_vs_baseline']:+.4f} |"
        )
    lines += [
        "",
        f"**Best shippable rule:** `{summary['best_shippable_rule']}` "
        f"(Δ={rules_block[summary['best_shippable_rule']]['delta_vs_baseline']:+.4f} vs baseline).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--predictions", type=Path, required=True,
                   help="predictions.jsonl from the run being counterfactual-ed.")
    p.add_argument("--mapping", type=Path, action="append", default=[],
                   help="Pickle mapping (chunk_id -> {text, metadata}). Repeatable.")
    p.add_argument("--corpus", type=Path, action="append", default=[],
                   help="JSONL corpus fallback. Repeatable.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory; created if missing.")
    p.add_argument("--proxy-threshold", type=float, default=0.30,
                   help="Token-overlap proxy threshold for the sentence router.")
    p.add_argument("--skip-source", action="append", default=["yargitay"],
                   help="Sources to exclude from snap candidates (repeatable).")
    p.add_argument("--bootstrap-n", type=int, default=1000,
                   help="Bootstrap resamples for the per-rule F1 CI.")
    args = p.parse_args(argv)

    if not args.mapping and not args.corpus:
        p.error("At least one of --mapping or --corpus is required.")

    predictions = load_predictions(args.predictions)
    logger.info("Loaded %d predictions from %s", len(predictions), args.predictions)

    chunk_lookup = load_chunk_lookup(args.mapping, args.corpus)
    if not chunk_lookup:
        raise RuntimeError("Empty chunk lookup — check --mapping/--corpus paths.")

    skip_sources = {s.lower() for s in args.skip_source if s}

    args.out.mkdir(parents=True, exist_ok=True)
    summary = evaluate_rules(
        predictions,
        chunk_lookup,
        skip_sources=skip_sources,
        proxy_threshold=args.proxy_threshold,
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
