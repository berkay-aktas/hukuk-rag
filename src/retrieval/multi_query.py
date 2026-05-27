"""Multi-query expansion: paraphrase the question, retrieve each, RRF-fuse.

A single phrasing of a legal question rarely covers every way the answer
might appear in the corpus. Asking the LLM for 3 paraphrases and running
each through retrieval, then RRF-merging the rankings, surfaces chunks the
original query missed without changing the corpus or the retrievers.

Implementation notes:

* Variants are generated greedy with a single LLM call (one prompt returns
  a numbered list). One LLM invocation per question, regardless of N.
* Each variant runs through the same retrievers as the original question
  and the resulting rankings are RRF-merged. The original question's
  retrieval is included in the fusion so we never do worse than baseline.
* Cost: one extra LLM call per question (~2-3s on Qwen-7B 4-bit, L4) plus
  N additional dense/sparse searches. Searches are cheap relative to
  generation, so total overhead is dominated by the one LLM call.

References:
* Langchain's MultiQueryRetriever (the pattern, not a code dependency).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.generation.llm import generate_answer

if TYPE_CHECKING:
    from src.generation.llm import LoadedLLM

logger = logging.getLogger(__name__)


MULTI_QUERY_SYSTEM_PROMPT = (
    "Sen Türk hukuku konusunda uzman bir asistansın. Sana verilen soruyu, "
    "aynı anlamı koruyacak şekilde farklı kelimelerle 3 farklı biçimde "
    "yeniden yaz. Hukuki terminolojiyi koru. Her satıra bir varyant yaz; "
    "numaralandırma kullan (1., 2., 3.). Açıklama, giriş veya kapanış cümlesi ekleme."
)

# Captures lines that start with a numeric prefix like "1." or "1)".
_VARIANT_LINE_RE = re.compile(r'^\s*(?:\d+[.)\-:]\s*)?(.+?)\s*$')


def generate_query_variants(
    llm: LoadedLLM,
    question: str,
    *,
    n: int = 3,
    max_new_tokens: int = 192,
    temperature: float = 0.0,
    repetition_penalty: float = 1.2,
) -> list[str]:
    """Return ``n`` paraphrases of ``question`` (plus the original).

    The returned list always starts with the original question so callers
    can safely use the list as the full set of retrieval queries: even if
    parsing fails, the original is preserved as a fallback.

    Greedy decoding keeps the output deterministic for reproducibility.
    Pass ``temperature>0`` if you want diverse paraphrases at the cost of
    repeatability.
    """
    raw = generate_answer(
        llm,
        system_prompt=MULTI_QUERY_SYSTEM_PROMPT,
        user_message=f"Soru: {question.strip()}\n\nVaryantlar:",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    variants: list[str] = []
    for line in raw.splitlines():
        m = _VARIANT_LINE_RE.match(line)
        if not m:
            continue
        cand = m.group(1).strip()
        if not cand:
            continue
        # Skip lines that are likely echoes of the system prompt or the
        # original question.
        if cand.lower() == question.strip().lower():
            continue
        variants.append(cand)
        if len(variants) >= n:
            break

    # Always include the original — defends against bad parses.
    return [question.strip()] + variants
