"""Hypothetical Document Embeddings (HyDE) for dense retrieval.

Standard dense retrieval embeds the question and searches for chunks whose
embeddings are close to it. HyDE inverts this: first ask the LLM to draft
the *answer* itself (a "hypothetical document"), then embed that draft as
the dense query. The answer-shaped embedding lives in the same neighborhood
as the true source passage, so the FAISS search has better surface match.

For Turkish legal QA the hypothetical answer prompt asks for a 2-3 sentence
statute-citing draft. Exactness doesn't matter — even a wrong draft puts
the embedding in the right "legal-answer" region of the encoder's space.

References:
* Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
  (HyDE, 2022). https://arxiv.org/abs/2212.10496

Cost: one extra LLM call per question (~2-4s on Qwen-7B 4-bit, L4). Best
combined with BM25 (which still uses the original question) so we don't
lose lexical-keyword matches when the LLM draft drifts from question
phrasing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.generation.llm import generate_answer

if TYPE_CHECKING:
    from src.generation.llm import LoadedLLM

logger = logging.getLogger(__name__)


HYDE_SYSTEM_PROMPT = (
    "Sen Türk hukuku alanında uzman bir asistansın. Sana verilen soruya "
    "verilebilecek 2-3 cümlelik kısa, kanun adı ve madde numarası içeren "
    "bir cevap taslağı yaz. Cevabını sanki gerçek bir kaynak metinmiş gibi "
    "yaz; doğruluğu garanti etmeye çalışma, soruya uygun türde bir cevap "
    "üret yeterli. Açıklama, giriş, tekrar veya gereksiz uyarı ekleme."
)


def generate_hypothetical_query(
    llm: LoadedLLM,
    question: str,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    repetition_penalty: float = 1.2,
) -> str:
    """Generate a hypothetical legal-style answer to ``question``.

    The output is used as the dense retrieval query (not shown to the user).
    Greedy decoding by default — the draft just needs to land in the right
    embedding neighborhood, not maximise diversity.

    Returns the draft as a single string, including the original question
    appended so the embedding mixes question and answer signals. Appending
    is empirically better than draft-only for short statutes where the
    question itself contains identifying terms (e.g., "kasten adam öldürme").
    """
    draft = generate_answer(
        llm,
        system_prompt=HYDE_SYSTEM_PROMPT,
        user_message=f"Soru: {question.strip()}",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    return f"{question.strip()} {draft.strip()}"
