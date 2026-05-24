"""Turkish RAG prompts, kept in one place so config swaps don't fan out across notebooks.

The default prompt is the one that produced the +49% baseline lift over the
unstandardized version in the C1 re-run (Mar 2026). Treat it as the calibration
constant: do not change it without re-running the C1 baseline.

For benchmarks where the gold answer is short (single citation, MCQ letter, etc.)
use ``SHORT_ANSWER_SYSTEM`` to suppress over-generation.
"""

from __future__ import annotations

from typing import Sequence

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = (
    "Sen bir Türk hukuku uzmanısın. Soruyu verilen bağlam paragraflarını "
    "kullanarak kısa ve öz şekilde yanıtla (2-3 cümle). İlgili kanun "
    "maddelerine atıfta bulun. Bağlamda bilgi yoksa "
    "'Bu konuda yeterli bilgi bulunamadı' de."
)

SHORT_ANSWER_SYSTEM = (
    "Sen bir Türk hukuku uzmanısın. Soruyu verilen bağlam paragraflarını "
    "kullanarak tek cümlede yanıtla. İlgili kanun ve madde numarasını "
    "belirt. Bağlamda bilgi yoksa 'Bu konuda yeterli bilgi bulunamadı' de. "
    "Gereksiz açıklama, giriş cümlesi veya tekrar ekleme."
)

CITATION_STRICT_SYSTEM = (
    "Sen bir Türk hukuku uzmanısın. Yanıtını YALNIZCA aşağıda numaralandırılmış "
    "[Kaynak N] paragraflarına dayandır.\n\n"
    "ZORUNLU KURALLAR:\n"
    "1. Her atıfta kanun adını VE madde numarasını AYNEN belirt "
    "(örn: 'Türk Ceza Kanunu Madde 81').\n"
    "2. Birden fazla kaynak ilgiliyse hepsini sentezle ve [Kaynak N] numarasıyla göster.\n"
    "3. Bağlam yetersizse 'Bu konuda yeterli bilgi bulunamadı' yaz; tahmin yürütme.\n\n"
    "Yanıt yapısı: ilk cümle doğrudan cevap; devamı kanun adı + madde + kısa "
    "açıklama. Gereksiz giriş cümlesi veya tekrar ekleme. Yalnızca Türkçe yanıt ver."
)

MCQ_SYSTEM = (
    "Sen Türk hukuku konusunda uzman bir asistansın. Sana bir soru ve "
    "şıklar verilecek; ayrıca ilgili kanun maddelerinden alıntılar sunulacak. "
    "Cevabını SADECE harf olarak ver. Format: 'Cevap: <X>' (X = A, B, C, D veya E)."
)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context_block(passages: Sequence[str], *, max_chars_per_passage: int = 1500) -> str:
    """Render retrieved passages as a numbered ``[Kaynak N]`` block.

    Args:
        passages: Iterable of passage texts in retrieval order.
        max_chars_per_passage: Truncate each passage to this length to keep
            the prompt within LLM context. 1500 chars ≈ 500 tokens.

    Returns:
        Newline-joined block ready to splice into a user message.
    """
    parts = []
    for i, p in enumerate(passages, 1):
        text = p.strip()
        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage].rstrip() + "…"
        parts.append(f"[Kaynak {i}]\n{text}")
    return "\n\n".join(parts)


def build_user_message(question: str, passages: Sequence[str], *, options: Sequence[str] | None = None) -> str:
    """Build the user message that will be appended after the system prompt.

    Args:
        question: The user's question (Turkish).
        passages: Retrieved passages, already ordered.
        options: Optional MCQ option list (A-E). When present, formats as
            ``A) ... B) ...`` so the system prompt's ``Cevap: X`` directive applies.

    Returns:
        A single string suitable for the user role in a chat template.
    """
    parts = [f"Soru: {question.strip()}"]
    if options:
        letters = ["A", "B", "C", "D", "E"]
        option_lines = [f"{letters[i]}) {opt}" for i, opt in enumerate(options) if i < 5]
        parts.append("Şıklar:\n" + "\n".join(option_lines))
    parts.append("Bağlam:\n" + format_context_block(passages))
    return "\n\n".join(parts)
