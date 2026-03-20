"""Evaluation metrics for retrieval and generation quality.

Retrieval: Recall@K, MRR, nDCG@K (binary relevance).
Generation: Exact Match, Token F1 (SQuAD-style), ROUGE-L.
Statistical testing: percentile bootstrap CIs, Wilcoxon signed-rank.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable

import numpy as np

from src.utils.turkish import normalize_turkish

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[list[str]],
    k: int,
) -> float:
    """Compute mean Recall@K across queries.

    Args:
        retrieved_ids: List of retrieved chunk ID lists per query.
        relevant_ids: List of relevant chunk ID lists per query.
        k: Cutoff rank.

    Returns:
        Mean Recall@K.
    """
    scores = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        if not relevant:
            continue
        top_k = set(retrieved[:k])
        rel_set = set(relevant)
        scores.append(len(top_k & rel_set) / len(rel_set))
    return float(np.mean(scores)) if scores else 0.0


def mrr(
    retrieved_ids: list[list[str]],
    relevant_ids: list[list[str]],
) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        retrieved_ids: List of retrieved chunk ID lists per query.
        relevant_ids: List of relevant chunk ID lists per query.

    Returns:
        MRR score.
    """
    rr_scores = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        rel_set = set(relevant)
        rr = 0.0
        for rank, chunk_id in enumerate(retrieved, 1):
            if chunk_id in rel_set:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def ndcg_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[list[str]],
    k: int = 10,
) -> float:
    """Compute mean nDCG@K with binary relevance.

    Uses binary relevance (1 if in relevant set, 0 otherwise). For graded
    relevance, a separate implementation would be needed.

    Args:
        retrieved_ids: List of retrieved chunk ID lists per query.
        relevant_ids: List of relevant chunk ID lists per query.
        k: Cutoff rank.

    Returns:
        Mean nDCG@K.
    """
    ndcg_scores = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        rel_set = set(relevant)
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, chunk_id in enumerate(retrieved[:k])
            if chunk_id in rel_set
        )
        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def retrieval_metrics(
    retrieved_ids: list[list[str]],
    relevant_ids: list[list[str]],
) -> dict[str, float]:
    """Compute all retrieval metrics.

    Args:
        retrieved_ids: List of retrieved chunk ID lists per query.
        relevant_ids: List of relevant chunk ID lists per query.

    Returns:
        Dict with Recall@5, Recall@10, MRR, nDCG@10.
    """
    return {
        "recall@5": recall_at_k(retrieved_ids, relevant_ids, k=5),
        "recall@10": recall_at_k(retrieved_ids, relevant_ids, k=10),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "ndcg@10": ndcg_at_k(retrieved_ids, relevant_ids, k=10),
    }


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def exact_match(predictions: list[str], references: list[str]) -> float:
    """Compute Exact Match score after Turkish normalization.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Fraction of exact matches.
    """
    matches = sum(
        1 for p, r in zip(predictions, references)
        if normalize_turkish(p) == normalize_turkish(r)
    )
    return matches / len(predictions) if predictions else 0.0


def token_f1(predictions: list[str], references: list[str]) -> float:
    """Compute SQuAD-style token-level F1 score.

    Uses multiset (Counter) overlap rather than set overlap so that
    duplicate tokens are counted correctly.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Mean token F1.
    """
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = Counter(normalize_turkish(pred).split())
        ref_tokens = Counter(normalize_turkish(ref).split())

        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue

        common = sum((pred_tokens & ref_tokens).values())
        if common == 0:
            f1_scores.append(0.0)
            continue

        precision = common / sum(pred_tokens.values())
        recall = common / sum(ref_tokens.values())
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def generation_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute all generation metrics.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Dict with EM, token F1, and ROUGE-L.
    """
    metrics = {
        "exact_match": exact_match(predictions, references),
        "token_f1": token_f1(predictions, references),
    }

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_scores = [
            scorer.score(normalize_turkish(ref), normalize_turkish(pred))["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        metrics["rouge_l"] = float(np.mean(rouge_scores))
    except ImportError:
        logger.warning("rouge_score not installed, skipping ROUGE-L")

    return metrics


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def bootstrap_ci(
    metric_fn: Callable,
    predictions: list,
    references: list,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute percentile bootstrap confidence intervals for a metric.

    Uses the percentile method (not BCa). For small or skewed samples
    the coverage may be approximate.

    Args:
        metric_fn: Function(predictions, references) -> float.
        predictions: List of predictions.
        references: List of references.
        n: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with mean, lower, upper.
    """
    rng = np.random.default_rng(seed)
    size = len(predictions)
    scores = []

    for _ in range(n):
        indices = rng.integers(0, size, size=size)
        boot_preds = [predictions[i] for i in indices]
        boot_refs = [references[i] for i in indices]
        scores.append(metric_fn(boot_preds, boot_refs))

    scores = sorted(scores)
    lower = scores[int(n * alpha / 2)]
    upper = scores[int(n * (1 - alpha / 2))]

    return {
        "mean": float(np.mean(scores)),
        "lower": float(lower),
        "upper": float(upper),
    }


def paired_significance_test(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, float]:
    """Wilcoxon signed-rank test between two systems.

    Assumes the distribution of pairwise differences is symmetric.
    For bounded or zero-inflated IR metrics this assumption may be
    violated; consider a permutation test as an alternative.

    Args:
        scores_a: Per-query scores for system A.
        scores_b: Per-query scores for system B.

    Returns:
        Dict with statistic and p_value.
    """
    from scipy.stats import wilcoxon

    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    if all(d == 0 for d in diffs):
        return {"statistic": 0.0, "p_value": 1.0}

    stat, p = wilcoxon(scores_a, scores_b)
    return {"statistic": float(stat), "p_value": float(p)}
