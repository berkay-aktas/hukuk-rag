"""LLM-as-Judge scorer matching the assignment's evaluation rubric.

The rubric specifies LLM-judged metrics for Faithfulness, Coherence, and
Relevancy across scenarios 1-3, plus optional LLM Judge for answer correctness
in scenarios 1 and 2. This module produces all four scores per question in a
single LLM call with structured JSON output, then aggregates over a benchmark
run.

Axes scored per question (each in [0, 1]):

* **correctness** — agreement between the predicted answer and the gold
  answer on the question's actual content (closest analogue to F1 from
  scenarios 1 and 2). When no gold is provided this axis is skipped.
* **faithfulness** — does the predicted answer assert only things the
  retrieved context supports? This is the scenario 1 G factor and the
  scenario 3 Faithfulness factor.
* **coherence** — fluent, well-structured Turkish prose with no
  contradictions. Scenario 3 only.
* **relevancy** — does the predicted answer address the user's question?
  Scenario 3.

Each call returns a JSON dict; we parse it defensively (an unparseable
output gets None scores so the aggregate is still computable). The judge is
prompted in Turkish for consistency with the answer language; instructions
demand JSON-only output to keep parsing reliable.

Designed to swap judge models without touching callers: pass any
:class:`~src.generation.llm.LoadedLLM` as ``judge_llm``. Default usage on
Colab is to reuse the same base Qwen2.5-7B-Instruct already loaded by the
RAG pipeline — same-model bias is real but acceptable for an internal
ranking, and we document the limitation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.generation.llm import LoadedLLM

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = (
    "Sen bir Türk hukuku değerlendirici uzmanısın. Görevin, bir RAG (kaynak "
    "gözeten yanıt) sisteminin ürettiği cevabı şu eksenlerde 0.00 ile 1.00 "
    "arası ondalık skorlarla değerlendirmektir:\n"
    "1. correctness  — Cevap, verilen 'altın cevap' ile içerik olarak ne kadar "
    "uyumlu? Aynı kanun ve madde numarasını mı işaret ediyor? Aynı temel "
    "kuralı mı söylüyor? (Altın cevap verilmemişse bu eksene null yaz.)\n"
    "2. faithfulness — Cevapta, kaynak metinde olmayan bir iddia var mı? "
    "1.00 = tümü desteklenmiş; 0.00 = uydurma.\n"
    "3. coherence    — Cevap akıcı, dilbilgisi açısından doğru ve mantıksal "
    "olarak tutarlı Türkçe mi?\n"
    "4. relevancy    — Cevap, sorulan soruyu gerçekten karşılıyor mu? "
    "Konu dışına çıkıyor mu?\n\n"
    "ÖNEMLİ: Yalnızca aşağıdaki formatta saf JSON döndür, açıklama veya ek "
    "metin EKLEME:\n"
    '{"correctness": 0.XX, "faithfulness": 0.XX, "coherence": 0.XX, '
    '"relevancy": 0.XX}'
)


_JSON_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)

# Axes we expect in the judge response. correctness can be None when the
# caller did not provide a gold answer.
_EXPECTED_AXES = ("correctness", "faithfulness", "coherence", "relevancy")


@dataclass
class JudgeScores:
    """One scored question."""

    question_id: str
    correctness: float | None
    faithfulness: float | None
    coherence: float | None
    relevancy: float | None
    raw_response: str = ""


def _parse_judge_output(raw: str) -> dict[str, float | None]:
    """Extract a JSON object of scores from the judge's free-text output.

    Strict JSON might not survive the LLM's natural drift toward explanations.
    We search for the first ``{...}`` block, parse it, then validate keys and
    coerce values into the [0, 1] range. Any axis that fails extraction comes
    back as None and is excluded from aggregates downstream.
    """
    if not raw:
        return {axis: None for axis in _EXPECTED_AXES}
    # Strip trailing junk: keep only the first balanced-looking JSON block.
    match = _JSON_RE.search(raw)
    if not match:
        return {axis: None for axis in _EXPECTED_AXES}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {axis: None for axis in _EXPECTED_AXES}
    out: dict[str, float | None] = {}
    for axis in _EXPECTED_AXES:
        v = obj.get(axis)
        if v is None:
            out[axis] = None
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            out[axis] = None
            continue
        out[axis] = max(0.0, min(1.0, f))
    return out


def _build_user_message(
    question: str,
    predicted_answer: str,
    *,
    gold_answer: str | None,
    retrieved_context: str,
    max_chars_per_field: int = 1800,
) -> str:
    """Pack the question, gold, prediction, and context into one message.

    We truncate each long field independently — better than truncating the
    whole message tail because the context tends to be much longer than the
    answer fields.
    """
    def _clip(s: str | None) -> str:
        if not s:
            return ""
        s = str(s).strip()
        return s[:max_chars_per_field] + (" …" if len(s) > max_chars_per_field else "")

    parts = [
        f"Soru:\n{_clip(question)}",
        f"Altın cevap:\n{_clip(gold_answer) if gold_answer else '(verilmedi — correctness eksenine null yaz)'}",
        f"Modelin ürettiği cevap:\n{_clip(predicted_answer)}",
        f"Kaynak (kullanılan bağlam):\n{_clip(retrieved_context)}",
        "JSON formatında 4 eksenli skorlarını ver.",
    ]
    return "\n\n".join(parts)


def score_one(
    judge_llm: LoadedLLM,
    *,
    question: str,
    predicted_answer: str,
    retrieved_context: str,
    gold_answer: str | None = None,
    question_id: str = "",
    max_new_tokens: int = 128,
) -> JudgeScores:
    """Score a single (question, prediction, context) triple via the judge LLM."""
    from src.generation.llm import generate_answer

    user_msg = _build_user_message(
        question, predicted_answer,
        gold_answer=gold_answer,
        retrieved_context=retrieved_context,
    )
    raw = generate_answer(
        judge_llm,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_message=user_msg,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        repetition_penalty=1.1,
    )
    parsed = _parse_judge_output(raw)
    return JudgeScores(
        question_id=question_id,
        correctness=parsed.get("correctness"),
        faithfulness=parsed.get("faithfulness"),
        coherence=parsed.get("coherence"),
        relevancy=parsed.get("relevancy"),
        raw_response=raw,
    )


def score_predictions(
    judge_llm: LoadedLLM,
    records: list[dict[str, Any]],
    *,
    chunk_lookup: dict[str, str] | None = None,
    context_field: str = "context_text",
    progress_every: int = 25,
) -> list[JudgeScores]:
    """Score every record in a benchmark predictions.jsonl.

    Each record must have ``question``, ``predicted_answer``; optionally
    ``gold_answer`` (correctness becomes null when missing), ``question_id``,
    and either ``retrieved_chunk_ids`` + a ``chunk_lookup`` dict OR a
    ``context_field`` value already populated. We avoid persisting the full
    retrieved text in predictions.jsonl for size reasons, so the chunk_lookup
    path is the common case.

    Returns a list of :class:`JudgeScores` in the same order as ``records``.
    """
    import time

    out: list[JudgeScores] = []
    started = time.time()
    for i, rec in enumerate(records):
        if context_field in rec and rec[context_field]:
            context_text = rec[context_field]
        elif chunk_lookup is not None:
            texts = [chunk_lookup.get(str(cid), "") for cid in (rec.get("retrieved_chunk_ids") or [])]
            context_text = "\n\n".join(t for t in texts if t)
        else:
            context_text = ""
        scores = score_one(
            judge_llm,
            question=rec.get("question", ""),
            predicted_answer=rec.get("predicted_answer", ""),
            retrieved_context=context_text,
            gold_answer=rec.get("gold_answer"),
            question_id=str(rec.get("question_id", i)),
        )
        out.append(scores)
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - started
            rate = (i + 1) / elapsed
            eta = (len(records) - (i + 1)) / rate if rate > 0 else 0
            logger.info(
                "scored %d/%d  (%.1fs elapsed, ~%.0fs ETA)",
                i + 1, len(records), elapsed, eta,
            )
    return out


def aggregate_scores(
    scores: list[JudgeScores],
    *,
    grouping_metadata: list[dict[str, Any] | None] | None = None,
    group_keys: tuple[str, ...] = ("domain", "difficulty"),
) -> dict[str, Any]:
    """Aggregate per-question scores into the headline numbers for a config.

    Returns mean of each axis (skipping None values) plus optional per-group
    breakdowns if ``grouping_metadata`` is supplied (typically the ``metadata``
    field from predictions.jsonl).
    """
    import numpy as np

    def _mean_nullable(values: list[float | None]) -> tuple[float | None, int]:
        clean = [v for v in values if v is not None]
        if not clean:
            return None, 0
        return float(np.mean(clean)), len(clean)

    headline: dict[str, Any] = {}
    for axis in _EXPECTED_AXES:
        values = [getattr(s, axis) for s in scores]
        mean, n = _mean_nullable(values)
        headline[axis] = {"mean": mean, "n_scored": n, "n_total": len(scores)}

    if grouping_metadata is None:
        return {"overall": headline}

    groups: dict[str, dict[str, list[float | None]]] = {}
    for s, md in zip(scores, grouping_metadata):
        md = md or {}
        for key in group_keys:
            bucket = str(md.get(key, "unknown"))
            slot = groups.setdefault(f"{key}={bucket}", {axis: [] for axis in _EXPECTED_AXES})
            for axis in _EXPECTED_AXES:
                slot[axis].append(getattr(s, axis))

    grouped: dict[str, dict[str, Any]] = {}
    for name, axes in sorted(groups.items()):
        block: dict[str, Any] = {"n": len(next(iter(axes.values())))}
        for axis, vals in axes.items():
            mean, n = _mean_nullable(vals)
            block[axis] = round(mean, 4) if mean is not None else None
        grouped[name] = block

    return {"overall": headline, "by_group": grouped}


def composite_score(
    aggregate: dict[str, Any],
    *,
    recall_at_k: float | None = None,
    semantic_similarity: float | None = None,
) -> dict[str, float | None]:
    """Compute the rubric's three composite scores from an aggregate dict.

    The aggregate dict is the output of :func:`aggregate_scores`. Recall@k and
    semantic similarity are passed separately because they aren't LLM-judged.

    Returns three values (Scenario 1, 2, 3); any that lack required inputs
    come back as None.
    """
    h = aggregate["overall"]
    correctness = (h.get("correctness") or {}).get("mean")
    faithfulness = (h.get("faithfulness") or {}).get("mean")
    coherence = (h.get("coherence") or {}).get("mean")
    relevancy = (h.get("relevancy") or {}).get("mean")

    scenario_1 = None
    if recall_at_k is not None and correctness is not None and faithfulness is not None:
        scenario_1 = 0.35 * recall_at_k + 0.40 * correctness + 0.25 * faithfulness

    scenario_2 = None
    if correctness is not None and semantic_similarity is not None:
        scenario_2 = 0.70 * correctness + 0.30 * semantic_similarity

    scenario_3 = None
    available = [v for v in (relevancy, faithfulness, coherence) if v is not None]
    if available:
        scenario_3 = sum(available) / len(available)

    return {
        "scenario_1": scenario_1,
        "scenario_2": scenario_2,
        "scenario_3": scenario_3,
    }
