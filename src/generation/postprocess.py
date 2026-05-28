"""Post-generation rewrites applied to the LLM's raw answer before scoring.

Three snap routers are shipped:

1. :func:`snap_to_context_sentence` — the original proxy-only router. Uses
   token-overlap fraction between the LLM answer and each candidate sentence
   as the routing signal. Cheap, no model required.
2. :func:`snap_to_context_sentence_nli` — NLI-gated router. Uses semantic
   similarity from a Turkish NLI sentence-transformer to decide whether the
   LLM is paraphrasing a specific context sentence. Closes false negatives
   (heavy paraphrase, low token overlap, high semantic similarity) that the
   proxy-only router misses.
3. :func:`snap_to_cited_madde` — citation-extraction router. Parses the LLM
   answer for explicit statute-and-article citations (e.g. ``TCK Madde 81``,
   ``Anayasa'nın 90. maddesi``). When the cited article is also present in
   the retrieved passages and the chunk's text matches the cited statute
   code, replaces the LLM answer with the verbatim article body. A grounded
   citation is a stronger signal of LLM intent than token overlap; this
   router takes precedence over (1) by default when both fire.

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
the signal. The citation router targets the same gap from a different angle:
it fires on explicit grounded citations regardless of paraphrase strength,
which makes it complementary to (1) and (2) rather than redundant.
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


# ---------------------------------------------------------------------------
# Citation-extraction snap router
# ---------------------------------------------------------------------------

# Canonical statute codes used in Turkish legal text. Each canonical key maps
# to the surface forms (abbreviations, full names, numbered law refs) that
# may appear in either the LLM's answer or a retrieved chunk's text. We need
# both for routing: the LLM's emission tells us *which* code is being cited,
# and the chunk's text tells us *whether* a retrieved passage matches that
# code — without the second check, a "Madde 5" mention in the answer could
# wrongly snap to any code's Article 5 in retrieval.
#
# Only codes that appear in our 3-corpus retrieval are listed. Anayasa is
# included even though the Constitution PDF is among the 7 missing statutes
# (HANDOFF.md) — when retrieval brings in an Anayasa reference indirectly
# (e.g. through Yargıtay decisions citing it) we still want the alias known.
_STATUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "TCK": (
        "TCK", "T.C.K.", "Türk Ceza Kanunu", "Türk Ceza Kanununun",
        "Ceza Kanunu", "5237 sayılı",
    ),
    "TBK": (
        "TBK", "T.B.K.", "Türk Borçlar Kanunu", "Türk Borçlar Kanununun",
        "Borçlar Kanunu", "6098 sayılı",
    ),
    "TMK": (
        "TMK", "T.M.K.", "Türk Medeni Kanunu", "Türk Medeni Kanununun",
        "Medeni Kanun", "4721 sayılı",
    ),
    "TTK": (
        "TTK", "T.T.K.", "Türk Ticaret Kanunu", "Türk Ticaret Kanununun",
        "Ticaret Kanunu", "6102 sayılı",
    ),
    "CMK": (
        "CMK", "C.M.K.", "Ceza Muhakemesi Kanunu", "Ceza Muhakemesi Kanununun",
        "5271 sayılı",
    ),
    "HMK": (
        "HMK", "H.M.K.", "Hukuk Muhakemeleri Kanunu", "Hukuk Muhakemeleri Kanununun",
        "6100 sayılı",
    ),
    "İYUK": (
        "İYUK", "IYUK", "İdari Yargılama Usulü Kanunu", "İdari Yargılama Usulü Kanununun",
        "2577 sayılı",
    ),
    "Anayasa": (
        "Anayasa", "Anayasa'nın", "Anayasanın",
        "Türkiye Cumhuriyeti Anayasası",
    ),
    "İK": (
        "İK", "İş Kanunu", "İş Kanununun", "4857 sayılı",
    ),
    "TKHK": (
        "TKHK", "Tüketicinin Korunması Hakkında Kanun", "6502 sayılı",
    ),
}

# Backref: alias surface (lowercased) → canonical code key.
_ALIAS_TO_CODE: dict[str, str] = {
    a.lower(): code for code, aliases in _STATUTE_ALIASES.items() for a in aliases
}

# Sort aliases by length descending so the longest match wins (e.g.,
# "Türk Ceza Kanununun" should match before "Ceza Kanunu" before "TCK").
_ALIAS_RE = re.compile(
    "|".join(
        re.escape(a)
        for a in sorted(_ALIAS_TO_CODE.keys(), key=len, reverse=True)
    ),
    re.IGNORECASE,
)

# Madde-N forward syntax: "Madde 81", "MADDE 81", "M. 81", "m. 81", "Md. 81".
# Captures the article number as group(1). A trailing /N or -X paragraph
# specifier is matched but discarded (we snap to the whole article, not the
# sub-paragraph — the chunked passage typically contains the full Madde).
_MADDE_FORWARD_RE = re.compile(
    r"\b(?:madde|m\.|md\.|MD\.)\s*(\d{1,4})(?:[/.\-](?:\d+|[A-Za-z]))?\b",
    re.IGNORECASE,
)

# Reverse syntax: "81. madde", "81. maddesi", "47/5. maddesi", "90'ıncı madde".
# Article number is group(1); we tolerate the genitive/ordinal suffix.
_MADDE_REVERSE_RE = re.compile(
    r"\b(\d{1,4})(?:[/.\-](?:\d+|[A-Za-z]))?[.'’]?\s*(?:inci|ıncı|uncu|üncü|nci|ncı)?\s*madde(?:sinde|sine|sinin|si|nin|ye|ne|de|dir)?\b",
    re.IGNORECASE,
)

# How wide a window (in characters) around a Madde-N match to scan for a
# statute alias. Tighter than a sentence — we want the alias adjacent to
# the article, not just somewhere nearby. ±60 covers the longest realistic
# verbatim form ("Türk Borçlar Kanununun 1023. maddesi") plus parenthesised
# abbreviations, without bleeding into adjacent legal clauses.
_CODE_ATTRIBUTION_WINDOW = 60

# False-positive guard: the alias "Anayasa" matches inside "Anayasa Mahkemesi"
# (Constitutional Court, a separate entity referenced by law 6216 — not the
# Constitution itself). We require the next short window after the alias
# match to NOT mention "Mahkeme" / "Mahkemesi" before accepting "Anayasa".
_AYM_DISAMBIGUATION_WINDOW = 25


def _resolve_alias_code(window: str, alias_global_start: int) -> str | None:
    """Walk the window's alias matches and return the canonical code, or None
    if attribution is ambiguous.

    "Ambiguous" means more than one *distinct* canonical code is mentioned in
    the window. A repeated mention of the same code is not ambiguous. The
    AYM-disambiguation guard runs here too: "Anayasa" followed by "Mahkeme"
    within ``_AYM_DISAMBIGUATION_WINDOW`` chars is rejected as a code match.
    """
    codes_found: list[str] = []
    for alias_m in _ALIAS_RE.finditer(window):
        matched = alias_m.group(0).lower()
        if matched == "anayasa":
            after = window[alias_m.end():alias_m.end() + _AYM_DISAMBIGUATION_WINDOW]
            if "mahkeme" in after.lower():
                continue
        code = _ALIAS_TO_CODE[matched]
        if not codes_found or codes_found[-1] != code:
            codes_found.append(code)

    distinct = set(codes_found)
    if len(distinct) == 1:
        return next(iter(distinct))
    return None  # zero matches OR multiple distinct codes both treated as "?"


def extract_citations(text: str) -> list[tuple[str, int]]:
    """Pull ``(canonical_code, article_no)`` pairs from LLM-produced text.

    Two-pass attribution:

    1. Every Madde-N match (forward or reverse syntax) gets a ±60 char
       window scanned for a recognizable statute alias. The window must
       contain exactly one *distinct* canonical code for the citation to
       be attributed; multi-code windows mark the citation as ``"?"``.
    2. If no alias is found in the window, the citation is left unattributed
       (code ``"?"``). If exactly one distinct code is present in the whole
       answer, all unattributed citations are reassigned to it. Otherwise
       they are dropped — bare ``"Madde 345"`` with multiple codes elsewhere
       is too ambiguous to ground safely.

    Returns deduplicated citations in first-occurrence order. An empty list
    means the answer either has no Madde-N mention or none could be
    attributed unambiguously.
    """
    if not text:
        return []

    # Collect (article_no, code, match_start) candidates from both syntaxes.
    raw: list[tuple[int, str, int]] = []
    for pattern in (_MADDE_FORWARD_RE, _MADDE_REVERSE_RE):
        for m in pattern.finditer(text):
            try:
                art_no = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if not (1 <= art_no <= 9999):
                continue
            win_start = max(0, m.start() - _CODE_ATTRIBUTION_WINDOW)
            win_end = min(len(text), m.end() + _CODE_ATTRIBUTION_WINDOW)
            code = _resolve_alias_code(text[win_start:win_end], win_start) or "?"
            raw.append((art_no, code, m.start()))

    if not raw:
        return []

    # Fallback attribution: if exactly one code appears anywhere in the
    # text, attribute unattributed citations to it. Otherwise drop them.
    distinct_codes = {code for _, code, _ in raw if code != "?"}
    if len(distinct_codes) == 1:
        sole = next(iter(distinct_codes))
        raw = [(n, sole if c == "?" else c, pos) for n, c, pos in raw]
    else:
        raw = [(n, c, pos) for n, c, pos in raw if c != "?"]

    # Dedupe (code, art_no) while preserving first-occurrence order.
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for n, c, _ in sorted(raw, key=lambda t: t[2]):
        key = (c, n)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _chunk_matches_code(chunk_text: str, code: str) -> bool:
    """True if the chunk text mentions any alias of the given canonical code.

    We scan the head of the chunk first (statute markers typically appear in
    the chunk's title or first paragraph), then fall back to the whole text.
    Lowercased substring match — robust to case and formatting variations.
    """
    if not chunk_text:
        return False
    aliases = _STATUTE_ALIASES.get(code, ())
    head = chunk_text[:1200].lower()
    for alias in aliases:
        if alias.lower() in head:
            return True
    if len(chunk_text) > 1200:
        tail = chunk_text[1200:].lower()
        for alias in aliases:
            if alias.lower() in tail:
                return True
    return False


# Regex used inside chunk text to locate the "Madde N" header. Note this
# differs from _MADDE_FORWARD_RE: chunk text uses the canonical "Madde N"
# form (no abbreviation variants), so we don't accept "m. N" here — that
# would match too aggressively in dense statute prose.
def _madde_header_re(art_no: int) -> re.Pattern[str]:
    return re.compile(rf"\bMadde\s+{art_no}\b", re.IGNORECASE)


# Mevzuat scraping bookkeeping that appears inside chunk text after the
# Madde header on some corpora (notably the Anayasa PDFs). Without stripping,
# the body extractor's first-N-sentences cap lands inside these metadata
# lines for short articles, returning bookkeeping instead of legal prose.
# Empirically: 11/16 citation-snap regressions on Phase 6 were caused by
# this pattern (Anayasa 15, 146, etc.). See EXPERIMENTS.md Experiment 11.
_MEVZUAT_METADATA_LINE = re.compile(
    r"^[ \t]*(?:Kaynak satır sayısı|doğrulama notu taşıyan satır sayısı)\s*[:：][^\n]*\n?",
    re.MULTILINE | re.IGNORECASE,
)

# Chunk continuation marker — appears in Yargıtay PDF chunks where one
# Madde spans multiple pages (e.g., "Madde 91 - Gözaltı (devam)"). When
# the body extractor lands on a continuation chunk, it misses the article
# opener that contains the gold-relevant fıkra. Detection is best-effort.
_CONTINUATION_MARKER = re.compile(r"\((?:devam|devamı)\)", re.IGNORECASE)


def _is_continuation_chunk(chunk_text: str, art_no: int) -> bool:
    """Heuristic: does ``Madde N`` in this chunk look like a continuation?"""
    m = _madde_header_re(art_no).search(chunk_text)
    if not m:
        return False
    # Look ±60 chars around the Madde header for the (devam) marker.
    window = chunk_text[max(0, m.start() - 60):min(len(chunk_text), m.end() + 60)]
    return bool(_CONTINUATION_MARKER.search(window))


def _extract_madde_body(chunk_text: str, art_no: int) -> str | None:
    """Return the body of Article ``art_no`` from a statute chunk.

    Locates ``Madde {art_no}`` in the chunk and extracts text from there
    up to the next ``Madde N`` marker (or end of chunk, whichever comes
    first). Trims long bodies to ~3 sentences / 500 chars so the snap
    doesn't dump an entire 1000-char chunk into the answer, but is large
    enough to absorb a one-line title + an opening prose sentence.

    Two pre-processing passes guard against known failure modes:

    1. Strip mevzuat metadata lines (``Kaynak satır sayısı: …``,
       ``doğrulama notu taşıyan satır sayısı: …``). Without this, short
       Anayasa articles have the bookkeeping show up as the "first
       sentence" of the body — see Experiment 11 in EXPERIMENTS.md.
    2. The 3-sentence/500-char cap is wider than the previous 2/400 so
       that an em-dash title sentence (``Madde 15 — Temel hak ve
       hürriyetlerin kullanılması…``) plus the actual prose opener
       fits in a single snap target.
    """
    chunk_text = _MEVZUAT_METADATA_LINE.sub("", chunk_text)

    m = _madde_header_re(art_no).search(chunk_text)
    if not m:
        return None

    start = m.start()
    rest = chunk_text[m.end():]
    next_madde = re.search(r"\bMadde\s+\d+\b", rest, re.IGNORECASE)
    end = m.end() + next_madde.start() if next_madde else len(chunk_text)
    body = chunk_text[start:end].strip()

    # Collapse multi-blank runs left by stripped metadata lines.
    body = re.sub(r"\n\s*\n+", "\n", body).strip()

    # Cap to first 3 sentences. One sentence is often just the em-dash
    # title (``Madde 15 — Temel hak ve hürriyetlerin durdurulması.``);
    # the gold-matching prose usually lives in the next 1-2 sentences.
    sents = _SENT_SPLIT.split(body)
    if len(sents) > 3:
        body = " ".join(sents[:3]).strip()
    if len(body) > 500:
        body = body[:500].rstrip() + "…"
    return body


def find_grounded_madde(
    code: str,
    art_no: int,
    retrieved_texts: list[str],
) -> str | None:
    """Find the verbatim body of ``(code, art_no)`` in retrieved passages.

    Returns the article body if a retrieved chunk both (a) matches the
    requested statute code and (b) contains a ``Madde {art_no}`` header.
    Returns ``None`` if no chunk satisfies both conditions — the citation is
    not grounded and we should not snap.

    Two-pass preference: when multiple chunks for the same Madde are in
    retrieval (common for long articles split across PDF pages), prefer the
    chunk WITHOUT a ``(devam)`` continuation marker — that one contains the
    article opener (fıkra 1), which is almost always the gold-relevant text.
    Fall back to continuation chunks only if no opener is in retrieval.

    The grounding check is what makes citation-snap safer than blind
    citation-to-text substitution: a hallucinated citation (LLM cites a
    statute that's not in retrieval) produces a ``None`` here and the LLM's
    original answer is preserved.
    """
    if not retrieved_texts or code == "?":
        return None

    matching = [t for t in retrieved_texts if _chunk_matches_code(t, code)]
    # First pass: skip continuation chunks ("Madde 91 ... (devam)")
    for chunk_text in matching:
        if _is_continuation_chunk(chunk_text, art_no):
            continue
        body = _extract_madde_body(chunk_text, art_no)
        if body:
            return body
    # Fallback: if every matching chunk is a continuation, accept one
    # rather than miss the snap entirely.
    for chunk_text in matching:
        body = _extract_madde_body(chunk_text, art_no)
        if body:
            return body
    return None


def snap_to_cited_madde(
    llm_answer: str,
    retrieved_texts: list[str],
) -> tuple[str, list[tuple[str, int]], bool]:
    """Replace the LLM answer with a grounded madde body when one is cited.

    Walks the citations extracted from ``llm_answer`` in occurrence order.
    The first citation that resolves to a grounded chunk wins — subsequent
    citations are still returned in the citation list (for logging) but do
    not trigger additional snaps.

    Returns ``(final_answer, extracted_citations, was_snapped)``. The
    citation list is returned even when no snap fired, which is useful for
    inspection: it tells us whether the LLM cited anything at all.
    """
    if not retrieved_texts:
        return llm_answer, [], False

    citations = extract_citations(llm_answer)
    if not citations:
        return llm_answer, [], False

    for code, art_no in citations:
        body = find_grounded_madde(code, art_no, retrieved_texts)
        if body:
            return body, citations, True

    return llm_answer, citations, False


# ---------------------------------------------------------------------------
# Unified router — picks among proxy-only / NLI-gated / citation snap.
# ---------------------------------------------------------------------------

# Precedence rules for combining citation-snap with sentence-snap. Used both
# at inference (RagPipeline) and in the offline counterfactual sweep.
CitationPrecedence = str  # "citation_first" | "sentence_first" | "citation_only" | "sentence_only"


def snap_route_decision(
    llm_answer: str,
    retrieved_texts: list[str],
    *,
    nli_scorer: NLIScorer | None = None,
    proxy_threshold: float = 0.30,
    sim_threshold: float = 0.65,
    prefilter_top_k: int = 5,
    top_k_sentences: int = 1,
    use_citation_snap: bool = False,
    citation_precedence: CitationPrecedence = "citation_first",
) -> tuple[str, dict[str, float], bool]:
    """Single entrypoint that picks among citation, proxy, or NLI snap routing.

    When ``use_citation_snap=True``, citation-snap may fire (or not) according
    to ``citation_precedence``:

    * ``"citation_first"`` — try citation-snap first; if it doesn't fire,
      fall through to the sentence router (proxy or NLI).
    * ``"sentence_first"`` — try the sentence router first; if it doesn't
      fire, fall through to citation-snap.
    * ``"citation_only"`` — try citation-snap and stop (no sentence router).
    * ``"sentence_only"`` — disable citation-snap entirely; equivalent to
      ``use_citation_snap=False``.

    When ``use_citation_snap=False`` (default), behaves identically to the
    previous proxy/NLI-only routing, preserving backward-compatible behaviour
    for the production C3 configuration.

    Returns ``(answer, signals, was_snapped)`` where ``signals`` contains
    whichever routing scores were actually computed — useful for the
    counterfactual script which wants to log all signals when sweeping.
    The signals dict may include ``proxy``, ``nli_sim``, ``citation_fired``,
    ``citations`` depending on which routers ran.
    """
    signals: dict[str, float] = {}

    def _run_sentence() -> tuple[str, bool]:
        if nli_scorer is None:
            ans, proxy, fired = snap_to_context_sentence(
                llm_answer,
                retrieved_texts,
                proxy_threshold=proxy_threshold,
                top_k_sentences=top_k_sentences,
            )
            signals["proxy"] = float(proxy)
            return ans, fired

        ans, sim, fired = snap_to_context_sentence_nli(
            llm_answer,
            retrieved_texts,
            nli_scorer,
            sim_threshold=sim_threshold,
            prefilter_top_k=prefilter_top_k,
            top_k_sentences=top_k_sentences,
        )
        signals["nli_sim"] = float(sim)
        return ans, fired

    def _run_citation() -> tuple[str, bool]:
        ans, cits, fired = snap_to_cited_madde(llm_answer, retrieved_texts)
        signals["citation_fired"] = 1.0 if fired else 0.0
        signals["citations_n"] = float(len(cits))
        return ans, fired

    if not use_citation_snap or citation_precedence == "sentence_only":
        ans, fired = _run_sentence()
        return ans, signals, fired

    if citation_precedence == "citation_only":
        ans, fired = _run_citation()
        return ans, signals, fired

    if citation_precedence == "citation_first":
        cit_ans, cit_fired = _run_citation()
        if cit_fired:
            return cit_ans, signals, True
        sent_ans, sent_fired = _run_sentence()
        return sent_ans, signals, sent_fired

    if citation_precedence == "sentence_first":
        sent_ans, sent_fired = _run_sentence()
        if sent_fired:
            return sent_ans, signals, True
        cit_ans, cit_fired = _run_citation()
        return cit_ans, signals, cit_fired

    raise ValueError(f"Unknown citation_precedence: {citation_precedence!r}")
