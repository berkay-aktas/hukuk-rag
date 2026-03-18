"""Evaluation metrics for retrieval and generation quality."""

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def recall_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[list[str]],
    k: int,
) -> float:
    """Compute Recall@K across queries.

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
    return np.mean(scores) if scores else 0.0


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
    return np.mean(rr_scores) if rr_scores else 0.0


def ndcg_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[list[str]],
    k: int = 10,
) -> float:
    """Compute nDCG@K.

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
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


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


def exact_match(predictions: list[str], references: list[str]) -> float:
    """Compute Exact Match score.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Fraction of exact matches.
    """
    matches = sum(
        1 for p, r in zip(predictions, references)
        if _normalize_turkish(p) == _normalize_turkish(r)
    )
    return matches / len(predictions) if predictions else 0.0


def token_f1(predictions: list[str], references: list[str]) -> float:
    """Compute token-level F1 score.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Mean token F1.
    """
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(_normalize_turkish(pred).split())
        ref_tokens = set(_normalize_turkish(ref).split())

        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue

        common = pred_tokens & ref_tokens
        if not common:
            f1_scores.append(0.0)
            continue

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return np.mean(f1_scores) if f1_scores else 0.0


def generation_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute all generation metrics.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Dict with EM, F1, ROUGE-L.
    """
    metrics = {
        "exact_match": exact_match(predictions, references),
        "token_f1": token_f1(predictions, references),
    }

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_scores = [
            scorer.score(_normalize_turkish(ref), _normalize_turkish(pred))["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        metrics["rouge_l"] = np.mean(rouge_scores)
    except ImportError:
        logger.warning("rouge_score not installed, skipping ROUGE-L")

    return metrics


def bootstrap_ci(
    metric_fn: Callable,
    predictions: list,
    references: list,
    n: int = 1000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute bootstrap confidence intervals for a metric.

    Args:
        metric_fn: Function(predictions, references) -> float.
        predictions: List of predictions.
        references: List of references.
        n: Number of bootstrap samples.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Dict with mean, lower, upper.
    """
    rng = np.random.RandomState(42)
    size = len(predictions)
    scores = []

    for _ in range(n):
        indices = rng.randint(0, size, size)
        boot_preds = [predictions[i] for i in indices]
        boot_refs = [references[i] for i in indices]
        scores.append(metric_fn(boot_preds, boot_refs))

    scores = sorted(scores)
    lower = scores[int(n * alpha / 2)]
    upper = scores[int(n * (1 - alpha / 2))]

    return {
        "mean": np.mean(scores),
        "lower": lower,
        "upper": upper,
    }


def paired_significance_test(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, float]:
    """Wilcoxon signed-rank test between two systems.

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


TURKISH_LOWER_MAP = str.maketrans("İIÖÜÇŞĞ", "iıöüçşğ")


def _normalize_turkish(text: str) -> str:
    """Normalize Turkish text for comparison.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    import re
    text = text.translate(TURKISH_LOWER_MAP).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
