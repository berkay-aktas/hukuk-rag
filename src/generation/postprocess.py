"""Post-generation rewrites applied to the LLM's raw answer before scoring.

Two snap routers are shipped:

1. :func:`snap_to_context_sentence` — the original proxy-only router. Uses
   token-overlap fraction between the LLM answer and each candidate sentence
   as the routing signal. Cheap, no model required.
2. :func:`snap_to_context_sentence_nli` — NLI-gated router. Uses semantic
   similarity from a Turkish NLI sentence-transformer to decide whether the
   LLM is paraphrasing a specific context sentence. Closes false negatives
   (heavy paraphrase, low token overlap, high semantic similarity) that the
   proxy-only router misses.

Why snap at all:

* Triage of the 3-corpus base-Qwen run showed 33% of questions have the gold
  content in the top-10 retrieved passages but the LLM still paraphrases away
  from it, losing token F1.
* When the LLM's answer is semantically close to a single context sentence,
  the LLM is paraphrasing it. The verbatim source scores higher against
  verbatim-statute gold than the paraphrase does.
* When the LLM is synthesizing across passages, snapping to one of them
  hurts. We keep the LLM output.

Measured lift on our 225q gold (proxy-only router):

============================================================  ======
config                                                             F1
============================================================  ======
3-corpus base + off-shelf reranker, no snap                   0.2235
3-corpus base + off-shelf reranker + snap @ proxy>=0.3        0.2386
oracle max(base, snap-k1, snap-k2)                            0.2861
============================================================  ======

The 0.30 token-overlap threshold dominated 0.40 and 0.50 in offline sweep.
The NLI-gated router targets the gap between 0.2386 and 0.2861 by snapping
on questions where the LLM paraphrased so heavily that token overlap missed
the signal.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from src.utils.turkish import normalize_turkish

if TYPE_CHECKING:
    from src.evaluation.nli import NLIScorer

# Period-ish + Turkish-uppercase start. Covers TR sentence boundaries while
# ignoring abbreviations and section markers used in statutes.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-ZĞÜŞİÖÇ0-9])')

# Drop very short or very long fragments — usually headers, captions, run-on
# legal preambles. Tuned for the chunk lengths in our corpora (mostly statute
# articles and Yargıtay decision snippets).
_MIN_SENT_LEN = 10
_MAX_SENT_LEN = 500


def context_sentences(retrieved_texts: list[str]) -> list[str]:
    """Flatten retrieved passages into a list of candidate sentences."""
    return [
        s.strip()
        for t in retrieved_texts
        for s in _SENT_SPLIT.split((t or "").strip())
        if _MIN_SENT_LEN <= len(s.strip()) <= _MAX_SENT_LEN
    ]


def snap_to_context_sentence(
    llm_answer: str,
    retrieved_texts: list[str],
    *,
    proxy_threshold: float = 0.30,
    top_k_sentences: int = 1,
) -> tuple[str, float, bool]:
    """If the LLM clearly paraphrased a context sentence, replace with verbatim.

    Computes the top-1 context sentence by token overlap with ``llm_answer``,
    normalised by ``len(llm_answer_tokens)`` (the proxy). When this proxy meets
    ``proxy_threshold`` we treat the LLM as having paraphrased that sentence
    and return the original verbatim form instead.

    Args:
        llm_answer: Raw LLM output for the question.
        retrieved_texts: Passages that were in the LLM's context window.
        proxy_threshold: Minimum overlap fraction to trigger the snap.
        top_k_sentences: Concatenate top-k sentences. Empirically k=1 wins;
            k=2 underperforms on our data because the second sentence drags
            in unrelated content.

    Returns:
        ``(final_answer, proxy_score, was_snapped)``.
    """
    if not retrieved_texts:
        return llm_answer, 0.0, False

    sentences = context_sentences(retrieved_texts)
    if not sentences:
        return llm_answer, 0.0, False

    anchor_tokens = Counter(normalize_turkish(llm_answer).split())
    anchor_size = max(1, sum(anchor_tokens.values()))

    scored = sorted(
        (
            (sum((anchor_tokens & Counter(normalize_turkish(s).split())).values()), s)
            for s in sentences
        ),
        key=lambda x: -x[0],
    )
    top_overlap, _ = scored[0]
    proxy = top_overlap / anchor_size

    if proxy < proxy_threshold:
        return llm_answer, proxy, False

    chosen = " ".join(s for _, s in scored[:top_k_sentences])
    return chosen, proxy, True


def snap_to_context_sentence_nli(
    llm_answer: str,
    retrieved_texts: list[str],
    nli_scorer: NLIScorer,
    *,
    sim_threshold: float = 0.65,
    prefilter_top_k: int = 5,
    top_k_sentences: int = 1,
) -> tuple[str, float, bool]:
    """NLI-gated variant: snap when semantic similarity exceeds a threshold.

    Token-overlap routing misses cases where the LLM heavily paraphrased a
    context sentence (low token overlap but the meaning is identical). NLI
    similarity catches those. Algorithm:

    1. Flatten retrieved passages into candidate sentences.
    2. Pre-filter to top-``prefilter_top_k`` by token overlap with the LLM
       answer (cheap, narrows the candidate space the NLI model has to score).
    3. For each candidate, compute cosine similarity between its NLI embedding
       and the LLM-answer embedding.
    4. If the best candidate's similarity meets ``sim_threshold``, snap to it
       (verbatim). Otherwise keep the LLM answer.

    The pre-filter is a quality optimization, not a correctness requirement:
    a context can have dozens of sentences, and scoring all of them with NLI
    is wasted compute when only a handful are plausibly close. Setting
    ``prefilter_top_k`` higher than the typical candidate count is a no-op
    (we keep all sentences); setting it too low can throw away high-NLI but
    low-overlap candidates, so default 5 is roomy.

    Args:
        llm_answer: Raw LLM output for the question.
        retrieved_texts: Passages that were in the LLM's context window.
        nli_scorer: Loaded :class:`~src.evaluation.nli.NLIScorer`.
        sim_threshold: Minimum cosine similarity to trigger the snap. Default
            0.65 is a reasonable starting point for the emrecan model and
            should be swept on the counterfactual eval per corpus / question
            distribution.
        prefilter_top_k: Number of candidates to score with NLI per question.
        top_k_sentences: Concatenate top-k by NLI sim if you want a 2-sentence
            snap. Empirically k=1 wins on our 225q gold.

    Returns:
        ``(final_answer, best_similarity, was_snapped)``.
    """
    if not retrieved_texts:
        return llm_answer, 0.0, False

    sentences = context_sentences(retrieved_texts)
    if not sentences:
        return llm_answer, 0.0, False

    # Cheap pre-filter — token-overlap top-K shortlist. Avoids running the
    # transformer on every sentence in long passages.
    anchor_tokens = Counter(normalize_turkish(llm_answer).split())
    scored_overlap = sorted(
        (
            (sum((anchor_tokens & Counter(normalize_turkish(s).split())).values()), s)
            for s in sentences
        ),
        key=lambda x: -x[0],
    )
    candidates = [s for _, s in scored_overlap[:prefilter_top_k]]
    if not candidates:
        return llm_answer, 0.0, False

    sims = nli_scorer.batch_similarity(llm_answer, candidates)
    order = np.argsort(-sims)  # descending
    best_idx = int(order[0])
    best_sim = float(sims[best_idx])

    if best_sim < sim_threshold:
        return llm_answer, best_sim, False

    chosen = " ".join(candidates[int(i)] for i in order[:top_k_sentences])
    return chosen, best_sim, True


def snap_route_decision(
    llm_answer: str,
    retrieved_texts: list[str],
    *,
    nli_scorer: NLIScorer | None = None,
    proxy_threshold: float = 0.30,
    sim_threshold: float = 0.65,
    prefilter_top_k: int = 5,
    top_k_sentences: int = 1,
) -> tuple[str, dict[str, float], bool]:
    """Single entrypoint that picks proxy-only or NLI-gated routing.

    When ``nli_scorer`` is None, behaves identically to
    :func:`snap_to_context_sentence` with the given ``proxy_threshold``.
    When ``nli_scorer`` is provided, uses :func:`snap_to_context_sentence_nli`.

    Returns ``(answer, signals, was_snapped)`` where ``signals`` contains
    whichever routing scores were actually computed — useful for the
    counterfactual script which wants to log both signals when sweeping.
    """
    if nli_scorer is None:
        answer, proxy, fired = snap_to_context_sentence(
            llm_answer,
            retrieved_texts,
            proxy_threshold=proxy_threshold,
            top_k_sentences=top_k_sentences,
        )
        return answer, {"proxy": float(proxy)}, fired

    answer, sim, fired = snap_to_context_sentence_nli(
        llm_answer,
        retrieved_texts,
        nli_scorer,
        sim_threshold=sim_threshold,
        prefilter_top_k=prefilter_top_k,
        top_k_sentences=top_k_sentences,
    )
    return answer, {"nli_sim": float(sim)}, fired
