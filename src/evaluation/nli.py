"""NLI / semantic-similarity scorer for snap routing and hallucination analysis.

Wraps a Turkish sentence-transformer fine-tuned on NLI + STS data so the
embedding space approximates entailment likelihood: pairs that entail each
other land closer than pairs that contradict.

The model used by default is ``emrecan/bert-base-turkish-cased-mean-nli-stsb-tr``
(110M params, mentioned in CLAUDE.md as the project's chosen NLI model).
Output is cosine similarity in roughly ``[0, 1]`` because both embeddings are
L2-normalized. We treat that scalar as a soft entailment proxy.

Two consumers:

* :mod:`src.generation.postprocess` — NLI-gated snap router. Replaces the
  proxy-only threshold (token overlap fraction) with a semantic-equivalence
  check between the LLM's answer and each candidate context sentence. This
  closes the 0.286 oracle ceiling that pure token-overlap routing leaves
  on the table.
* :mod:`scripts.hallucination_analysis` — sentence-level faithfulness. For
  each sentence in the LLM's answer, compute max similarity to any context
  sentence; flag low-similarity sentences as likely hallucinations. This
  covers the assignment's mandatory NLI hallucination analysis deliverable.

Loading is lazy (sentence-transformers is heavy). The scorer is created once
per process and reused; ``encode`` is batched and L2-normalized internally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"


@dataclass
class NLIScorer:
    """Wraps a sentence-transformer for fast semantic similarity scoring.

    Cosine similarity between L2-normalized embeddings is in [-1, 1] but
    for NLI-fine-tuned encoders on Turkish text it sits roughly in [0, 1]
    in practice (related-language pairs rarely produce strong anti-alignment).

    Use :func:`load_nli_scorer` to construct.
    """

    model: SentenceTransformer
    model_id: str
    encode_batch_size: int = 64

    def similarity(self, anchor: str, candidate: str) -> float:
        """Cosine similarity between a single (anchor, candidate) pair."""
        sims = self.batch_similarity(anchor, [candidate])
        return float(sims[0])

    def batch_similarity(self, anchor: str, candidates: list[str]) -> np.ndarray:
        """One anchor vs many candidates — the snap-router hot path.

        Returns a 1-D array of length ``len(candidates)``.
        """
        if not candidates:
            return np.zeros(0, dtype=np.float32)
        anchor_emb = self.model.encode(
            [anchor],
            normalize_embeddings=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        cand_embs = self.model.encode(
            candidates,
            normalize_embeddings=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        # Cosine sim reduces to dot product on normalized vectors.
        return (cand_embs @ anchor_emb.T).reshape(-1).astype(np.float32)

    def pairwise_similarity(
        self,
        texts_a: list[str],
        texts_b: list[str],
    ) -> np.ndarray:
        """Full ``[len(texts_a), len(texts_b)]`` similarity matrix.

        Used by hallucination analysis: for each LLM-answer sentence, compute
        similarity to every context sentence, then take the row-wise max as
        the per-sentence faithfulness score.
        """
        if not texts_a or not texts_b:
            return np.zeros((len(texts_a), len(texts_b)), dtype=np.float32)
        emb_a = self.model.encode(
            texts_a,
            normalize_embeddings=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        emb_b = self.model.encode(
            texts_b,
            normalize_embeddings=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        return (emb_a @ emb_b.T).astype(np.float32)


def load_nli_scorer(
    model_id: str = DEFAULT_NLI_MODEL,
    *,
    device: str | None = None,
    encode_batch_size: int = 64,
) -> NLIScorer:
    """Load the Turkish NLI sentence-transformer.

    Args:
        model_id: HuggingFace model id. Defaults to the project's chosen model.
        device: ``"cuda"``, ``"cpu"``, or ``None`` (auto). On Colab L4 the
            model runs in ~50ms per batch of 64 sentences; on Mac CPU it's
            usable but ~5x slower — fine for the 225-question counterfactual.
        encode_batch_size: Sentence batch size for the underlying transformer.

    Returns:
        A loaded :class:`NLIScorer` ready for ``.similarity`` / ``.batch_similarity``.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading NLI scorer: %s (device=%s)", model_id, device or "auto")
    model = SentenceTransformer(model_id, device=device)
    return NLIScorer(model=model, model_id=model_id, encode_batch_size=encode_batch_size)


def smoke_test(scorer: NLIScorer | None = None) -> dict[str, float]:
    """Quick sanity check that the model orders entailment > contradiction.

    Returns a dict of scores for inspection. The entailing pair should score
    visibly higher than the contradicting one; if they're close, the model
    didn't load with the NLI head correctly.
    """
    if scorer is None:
        scorer = load_nli_scorer()

    premise = "Hırsızlık suçunda fail bir yıldan üç yıla kadar hapis cezası ile cezalandırılır."
    entail = "Hırsızlık eden kişiye verilen ceza bir ila üç yıl hapistir."
    contradict = "Hırsızlık suçunun cezası yoktur, fail serbest bırakılır."
    unrelated = "Türkiye'nin başkenti Ankara'dır."

    sims = scorer.batch_similarity(premise, [entail, contradict, unrelated])
    result = {
        "entail": float(sims[0]),
        "contradict": float(sims[1]),
        "unrelated": float(sims[2]),
    }
    logger.info("NLI smoke test: %s", result)
    return result
