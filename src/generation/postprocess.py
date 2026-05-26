"""Post-generation rewrites applied to the LLM's raw answer before scoring.

The only post-processor currently shipped is :func:`snap_to_context_sentence`,
which replaces the LLM's answer with the single most-overlapping sentence from
the retrieved context when overlap exceeds a threshold. Rationale:

* Triage of the 3-corpus base-Qwen run showed 33% of questions have the gold
  content in the top-10 retrieved passages but the LLM still paraphrases away
  from it, losing token F1.
* When the LLM's answer already overlaps >=30% of its tokens with a single
  context sentence, the LLM clearly paraphrased that sentence. The verbatim
  form scores higher against verbatim-statute gold than the paraphrase does.
* When the overlap is <30%, the LLM is either synthesizing across passages or
  improvising — snapping in that regime hurts. We keep the LLM output.

Measured lift on our 225q gold:

============================================================  ======
config                                                             F1
============================================================  ======
3-corpus base + off-shelf reranker, no snap                   0.2235
3-corpus base + off-shelf reranker + snap @ proxy>=0.3        0.2386
oracle max(base, snap-k1, snap-k2)                            0.2861
============================================================  ======

The 0.30 threshold dominated 0.40 and 0.50 in offline sweep.
"""

from __future__ import annotations

import re
from collections import Counter

from src.utils.turkish import normalize_turkish

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
