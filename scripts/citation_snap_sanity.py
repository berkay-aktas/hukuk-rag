"""Sanity checks for the citation-snap router.

Run from repo root::

    python scripts/citation_snap_sanity.py

Validates:

1. ``extract_citations`` handles the seven syntactic patterns observed in
   live LLM outputs (forward "Madde N", reverse "N. maddesi", parenthesised
   "(TMK 1025)", verbatim header "MADDE 124-", genitive "Anayasa'nın 90. maddesi",
   numeric law refs "6216 sayılı ... 47/5. maddesi", and ambiguous bare
   "Madde N" with no nearby code).
2. ``find_grounded_madde`` returns the verbatim body when the chunk matches
   both the cited code and article, and ``None`` otherwise.
3. ``snap_to_cited_madde`` end-to-end: snaps on grounded citations, keeps the
   LLM answer when citations are not grounded or absent.
4. ``snap_route_decision`` honours each precedence rule.

Prints "PASS" / "FAIL" per check; exits non-zero on any failure so it can
gate a future CI hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.postprocess import (
    extract_citations,
    find_grounded_madde,
    snap_route_decision,
    snap_to_cited_madde,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures derived from real phase6 14B outputs (see HANDOFF)
# ---------------------------------------------------------------------------

TCK_81_CHUNK = (
    "Türk Ceza Kanunu — Kişiye Karşı Suçlar — Birinci Bölüm — Hayata Karşı Suçlar\n"
    "Madde 81- (1) Bir insanı kasten öldüren kişi, müebbet hapis cezası "
    "ile cezalandırılır.\n"
    "Madde 82- (1) Kasten öldürme suçunun; a) Tasarlayarak, b) Canavarca "
    "hisle veya eziyet çektirerek işlenmesi hâlinde, kişi ağırlaştırılmış "
    "müebbet hapis cezası ile cezalandırılır."
)

TBK_299_CHUNK = (
    "Türk Borçlar Kanunu — Kira sözleşmesi\n"
    "Madde 299- Kira sözleşmesi, kiraya verenin bir şeyin kullanılmasını "
    "veya kullanmayla birlikte ondan yararlanılmasını kiracıya bırakmayı, "
    "kiracının da buna karşılık kararlaştırılan kira bedelini ödemeyi "
    "üstlendiği sözleşmedir."
)

TTK_124_CHUNK = (
    "Türk Ticaret Kanunu — İkinci Kitap — Ticaret Şirketleri\n"
    "MADDE 124- (1) Ticaret şirketleri; kollektif, komandit, anonim, "
    "limited ve kooperatif şirketlerden ibarettir. (2) Bu Kanunda, "
    "kollektif ile komandit şirket şahıs..."
)

AYM_90_CHUNK = (
    "Türkiye Cumhuriyeti Anayasası — Madde 90- Türkiye Cumhuriyeti adına "
    "yabancı devletlerle ve milletlerarası kuruluşlarla yapılacak "
    "andlaşmaların onaylanması, Türkiye Büyük Millet Meclisinin "
    "onaylamayı bir kanunla uygun bulmasına bağlıdır."
)

UNRELATED_YARGITAY = (
    "Yargıtay 9. Hukuk Dairesi 2023/4567 E., 2024/123 K. — Davacı işveren "
    "iş sözleşmesinin haklı nedenle feshini ileri sürmüştür. Yerel mahkeme "
    "kararı bozulmuştur."
)


def _check(name: str, condition: bool, *, details: str = "") -> bool:
    """Print PASS/FAIL and return the boolean for outer aggregation."""
    tag = "PASS" if condition else "FAIL"
    print(f"  [{tag}] {name}")
    if not condition and details:
        print(f"         {details}")
    return condition


def test_extract_citations() -> list[bool]:
    print("\n== extract_citations ==")
    results = []

    # Forward syntax: "TCK Madde 81"
    out = extract_citations("TCK Madde 81'e göre kasten öldürmenin cezası müebbet hapistir.")
    results.append(_check("forward 'TCK Madde 81'", out == [("TCK", 81)], details=f"got {out}"))

    # Reverse syntax: "Anayasa'nın 90. maddesi"
    out = extract_citations("Anayasa'nın 90. maddesi gereği milletlerarası andlaşmalar kanun hükmündedir.")
    results.append(_check("reverse 'Anayasa\\'nın 90. maddesi'", out == [("Anayasa", 90)], details=f"got {out}"))

    # Genitive long form: "Türk Medeni Kanununun 1023. maddesi"
    out = extract_citations("Türk Medeni Kanununun 1023. maddesi tasarruf işlemlerini düzenler.")
    results.append(_check("'Türk Medeni Kanununun 1023. maddesi'", out == [("TMK", 1023)], details=f"got {out}"))

    # Paragraph specifier dropped: "Madde 47/5"
    out = extract_citations("6216 sayılı Anayasa Mahkemesinin... 47/5. maddesi 30 gün öngörür.")
    # 6216 is not in our alias map; fallback attribution should fail to attribute
    # (no recognized code in window) and the citation should be dropped — UNLESS
    # there's also exactly one recognized code elsewhere. Here there's none, so
    # we expect an empty result.
    results.append(_check(
        "'6216 sayılı ... 47/5. maddesi' drops because 6216 is not aliased",
        out == [],
        details=f"got {out}",
    ))

    # Parenthesised abbreviation: "(TMK 1025)" — bare "TMK" + bare 1025, no Madde word.
    # extract_citations requires a Madde token; bare "TMK 1025" alone should NOT match.
    # The full phrase "Türk Medeni Kanunu (TMK) Madde 1025'e göre" does match.
    out = extract_citations("Türk Medeni Kanunu (TMK) Madde 1025'e göre devir hükmü uygulanır.")
    results.append(_check("'TMK Madde 1025'", out == [("TMK", 1025)], details=f"got {out}"))

    # Verbatim header copy from retrieval: "MADDE 124- (1) ..."
    out = extract_citations("MADDE 124- (1) Ticaret şirketleri; kollektif, komandit, anonim... TTK gereği.")
    results.append(_check(
        "verbatim 'MADDE 124-' with TTK in same answer",
        out == [("TTK", 124)],
        details=f"got {out}",
    ))

    # Bare Madde with no nearby code AND no unique code in whole text — should be dropped.
    out = extract_citations("İstinaf süresi iki haftadır (MADDE 345).")
    results.append(_check(
        "ambiguous bare 'MADDE 345' (no code anywhere) returns []",
        out == [],
        details=f"got {out}",
    ))

    # Multi-citation answer: extract all unique pairs
    out = extract_citations(
        "TBK Madde 307 ve TBK Madde 308 ile TBK Madde 320 birlikte değerlendirilmelidir."
    )
    results.append(_check(
        "multi-citation 'TBK Madde 307/308/320'",
        out == [("TBK", 307), ("TBK", 308), ("TBK", 320)],
        details=f"got {out}",
    ))

    # No citation
    out = extract_citations("Bu konuda yeterli bilgi bulunamadı.")
    results.append(_check("no-citation answer returns []", out == [], details=f"got {out}"))

    # Empty input
    results.append(_check("empty string returns []", extract_citations("") == []))

    # Multiple distinct codes within the attribution window → "?" → drop.
    # Window is ±60 chars, so both codes must be inside that span for this
    # to trigger "?". Below: "TCK ... TBK" are both within 60 of Madde 5.
    out = extract_citations("TCK alanı düzenler. TBK farklı kanun olarak Madde 5 düzenlemesini içerir.")
    results.append(_check(
        "multi-code in attribution window → '?' → fallback drops",
        out == [],
        details=f"got {out}",
    ))

    # Single code in text, bare Madde fallback should ATTRIBUTE to it
    out = extract_citations("TCK kapsamında değerlendirildiğinde, Madde 5 farklı şekilde uygulanır.")
    results.append(_check(
        "bare 'Madde 5' with sole code 'TCK' attributes to TCK",
        out == [("TCK", 5)],
        details=f"got {out}",
    ))

    return results


def test_find_grounded_madde() -> list[bool]:
    print("\n== find_grounded_madde ==")
    results = []

    # Happy path: TCK 81 cited, TCK chunk in retrieval
    body = find_grounded_madde("TCK", 81, [TCK_81_CHUNK, UNRELATED_YARGITAY])
    results.append(_check(
        "TCK Madde 81 grounded in TCK chunk",
        body is not None and "müebbet hapis cezası" in body,
        details=f"body={body!r}",
    ))

    # Code mismatch: TBK cited but only TCK in retrieval — must NOT snap
    body = find_grounded_madde("TBK", 81, [TCK_81_CHUNK, UNRELATED_YARGITAY])
    results.append(_check(
        "TBK Madde 81 with only TCK chunk in retrieval returns None",
        body is None,
        details=f"body={body!r}",
    ))

    # Madde not in chunk: TCK 999 cited, TCK chunk has only 81-82
    body = find_grounded_madde("TCK", 999, [TCK_81_CHUNK])
    results.append(_check(
        "TCK Madde 999 (not in chunk) returns None",
        body is None,
        details=f"body={body!r}",
    ))

    # Unattributed code '?' returns None
    results.append(_check(
        "code='?' returns None",
        find_grounded_madde("?", 81, [TCK_81_CHUNK]) is None,
    ))

    # Empty retrieval returns None
    results.append(_check(
        "empty retrieval returns None",
        find_grounded_madde("TCK", 81, []) is None,
    ))

    # Body extraction stops at the next Madde header
    body = find_grounded_madde("TCK", 81, [TCK_81_CHUNK])
    results.append(_check(
        "extracted body does not include Madde 82",
        body is not None and "Madde 82" not in body,
        details=f"body={body!r}",
    ))

    # Anayasa lookup
    body = find_grounded_madde("Anayasa", 90, [AYM_90_CHUNK])
    results.append(_check(
        "Anayasa Madde 90 grounded",
        body is not None and "milletlerarası" in body.lower(),
        details=f"body={body!r}",
    ))

    return results


def test_snap_to_cited_madde() -> list[bool]:
    print("\n== snap_to_cited_madde ==")
    results = []

    llm_answer = "TCK Madde 81'e göre kasten öldürmenin cezası müebbet hapistir."
    snapped, cits, fired = snap_to_cited_madde(llm_answer, [TCK_81_CHUNK])
    results.append(_check(
        "grounded citation fires snap",
        fired and "müebbet hapis cezası" in snapped and cits == [("TCK", 81)],
        details=f"snapped={snapped!r}, cits={cits}",
    ))

    # Non-grounded citation (LLM cites TBK 81 but only TCK in retrieval)
    snapped, cits, fired = snap_to_cited_madde(
        "TBK Madde 81 düzenlemesi gereği...", [TCK_81_CHUNK, UNRELATED_YARGITAY],
    )
    results.append(_check(
        "ungrounded citation keeps LLM answer",
        not fired and snapped == "TBK Madde 81 düzenlemesi gereği..." and cits == [("TBK", 81)],
        details=f"snapped={snapped!r}, cits={cits}",
    ))

    # No citation in answer
    snapped, cits, fired = snap_to_cited_madde(
        "Bu konuda yeterli bilgi bulunamadı.", [TCK_81_CHUNK],
    )
    results.append(_check(
        "no citation -> no snap, citations empty",
        not fired and cits == [],
    ))

    # Multi-citation, same code: first that grounds wins. TCK Madde 999
    # doesn't exist in retrieval, TCK Madde 81 does → snap fires on 81.
    snapped, cits, fired = snap_to_cited_madde(
        "TCK Madde 999 ve TCK Madde 81 birlikte değerlendirilmelidir.",
        [TCK_81_CHUNK],
    )
    results.append(_check(
        "multi-citation same code: TCK 999 misses, TCK 81 hits -> snap to TCK 81",
        fired and "müebbet hapis cezası" in snapped,
        details=f"snapped={snapped[:120]!r}, cits={cits}",
    ))

    # Empty retrieved_texts
    snapped, cits, fired = snap_to_cited_madde("TCK Madde 81", [])
    results.append(_check(
        "empty retrieval -> no snap",
        not fired and snapped == "TCK Madde 81",
    ))

    return results


def test_snap_route_decision() -> list[bool]:
    print("\n== snap_route_decision (precedence) ==")
    results = []

    # Set up: LLM cites TCK 81 verbatim-ish AND has high overlap with a sentence
    # in the chunk — both routers would fire. Verify precedence.
    llm = "TCK Madde 81'e göre bir insanı kasten öldüren kişi müebbet hapis cezası ile cezalandırılır."
    texts = [TCK_81_CHUNK]

    # citation_first (default): citation-snap wins
    ans, sig, fired = snap_route_decision(
        llm, texts, use_citation_snap=True, citation_precedence="citation_first",
    )
    results.append(_check(
        "citation_first: citation-snap wins when both could fire",
        fired and sig.get("citation_fired", 0.0) > 0.5 and "müebbet hapis cezası" in ans,
        details=f"signals={sig}",
    ))

    # sentence_first: sentence-snap wins
    ans, sig, fired = snap_route_decision(
        llm, texts, use_citation_snap=True, citation_precedence="sentence_first",
        proxy_threshold=0.10,
    )
    results.append(_check(
        "sentence_first: sentence-snap wins when both could fire",
        fired and sig.get("proxy", 0.0) >= 0.10,
        details=f"signals={sig}, ans={ans[:100]!r}",
    ))

    # citation_only: doesn't fall through to sentence on miss
    ans, sig, fired = snap_route_decision(
        "Bu konuda yeterli bilgi yok.", texts,
        use_citation_snap=True, citation_precedence="citation_only",
    )
    results.append(_check(
        "citation_only: no fallback to sentence",
        not fired,
        details=f"signals={sig}, ans={ans[:100]!r}",
    ))

    # sentence_only: equivalent to use_citation_snap=False
    ans, sig, fired = snap_route_decision(
        llm, texts, use_citation_snap=True, citation_precedence="sentence_only",
        proxy_threshold=0.10,
    )
    results.append(_check(
        "sentence_only: citation router does not run",
        "citation_fired" not in sig,
        details=f"signals={sig}",
    ))

    # use_citation_snap=False: only sentence router runs
    ans, sig, fired = snap_route_decision(
        llm, texts, use_citation_snap=False, proxy_threshold=0.10,
    )
    results.append(_check(
        "use_citation_snap=False: backward-compat (sentence only)",
        "citation_fired" not in sig,
        details=f"signals={sig}",
    ))

    return results


def main() -> int:
    print("Citation-snap sanity suite")
    all_results = []
    all_results += test_extract_citations()
    all_results += test_find_grounded_madde()
    all_results += test_snap_to_cited_madde()
    all_results += test_snap_route_decision()
    passed = sum(all_results)
    total = len(all_results)
    print(f"\n{passed}/{total} checks passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
