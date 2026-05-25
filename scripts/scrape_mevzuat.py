"""Download major Turkish legal statutes (Mevzuat) as PDFs from mevzuat.gov.tr.

Each statute on mevzuat.gov.tr has a stable PDF URL of the form::

    https://www.mevzuat.gov.tr/MevzuatMetin/{tertip}.{tur}.{no}.pdf

Where:

- ``tertip`` is the hierarchy version (most modern statutes use ``1`` or ``5``).
- ``tur`` is the type code (``5`` for Kanun, ``1`` for Anayasa, etc.).
- ``no`` is the statute number (e.g. 5237 for TCK).

The downloader tries the URL with polite delays between requests, saves PDFs to
the output directory, and writes a manifest of successes/failures. PDFs are
text-extractable (not scanned images), so ``pypdf`` parses them cleanly.

Usage::

    python scripts/scrape_mevzuat.py --out data/external/mevzuat/raw/
    # then ingest:
    python -m hukuk_rag ingest --docs data/external/mevzuat/raw/ --out data/external/mevzuat/indexes/

Run time: ~1-2 minutes for ~50 PDFs (network-bound, ~1s/file with polite delay).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
BASE_URL = "https://www.mevzuat.gov.tr/MevzuatMetin"


@dataclass(frozen=True)
class Statute:
    """One statute to download. (name, mevzuat_no, tur, tertip)."""

    name: str
    no: int
    tur: int = 5      # Kanun (Law) — default
    tertip: int = 1   # Most modern laws use tertip 1


# Major Turkish legal codes. Source: mevzuat.gov.tr — these are the canonical
# codes most likely to appear in a Turkish legal QA benchmark.
STATUTES: list[Statute] = [
    # Constitution
    Statute("anayasa", 2709, tur=1, tertip=5),

    # Criminal
    Statute("turk_ceza_kanunu_5237", 5237),
    Statute("ceza_muhakemesi_kanunu_5271", 5271),
    Statute("ceza_ve_guvenlik_tedbirleri_5275", 5275),
    Statute("kabahatler_kanunu_5326", 5326),

    # Civil
    Statute("turk_medeni_kanunu_4721", 4721),
    Statute("turk_borclar_kanunu_6098", 6098),
    Statute("hukuk_muhakemeleri_kanunu_6100", 6100),
    Statute("icra_iflas_kanunu_2004", 2004),

    # Commercial
    Statute("turk_ticaret_kanunu_6102", 6102),
    Statute("sermaye_piyasasi_kanunu_6362", 6362),
    Statute("bankacilik_kanunu_5411", 5411),
    Statute("rekabetin_korunmasi_4054", 4054),

    # Labor / Social
    Statute("is_kanunu_4857", 4857),
    Statute("is_mahkemeleri_7036", 7036),
    Statute("sosyal_sigortalar_5510", 5510),
    Statute("isssizlik_sigortasi_4447", 4447),
    Statute("sendikalar_toplu_is_sozlesmesi_6356", 6356),

    # Administrative
    Statute("idari_yargilama_usulu_2577", 2577),
    Statute("danistay_kanunu_2575", 2575),
    Statute("devlet_memurlari_657", 657, tertip=5),
    Statute("dilekce_hakkinin_kullanilmasi_3071", 3071),

    # Tax / Finance
    Statute("vergi_usul_213", 213, tertip=5),
    Statute("gelir_vergisi_193", 193, tertip=5),
    Statute("kurumlar_vergisi_5520", 5520),
    Statute("katma_deger_vergisi_3065", 3065),
    Statute("amme_alacaklari_6183", 6183, tertip=3),
    Statute("ozel_tuketim_vergisi_4760", 4760),
    Statute("damga_vergisi_488", 488, tertip=5),
    Statute("kamu_mali_yonetimi_5018", 5018),

    # Consumer / Internet / Data
    Statute("tuketicinin_korunmasi_6502", 6502),
    Statute("internet_yayinlari_5651", 5651),
    Statute("kvkk_6698", 6698),
    Statute("elektronik_imza_5070", 5070),

    # Property / Real Estate
    Statute("kat_mulkiyeti_634", 634, tertip=5),
    Statute("kamulastirma_2942", 2942),
    Statute("kentsel_donusum_6306", 6306),
    Statute("tapu_2644", 2644, tertip=3),

    # Family / Inheritance (largely covered by TMK above)
    Statute("nufus_hizmetleri_5490", 5490),

    # Procedure / Legal Profession
    Statute("avukatlik_1136", 1136, tertip=5),
    Statute("noterlik_1512", 1512, tertip=5),
    Statute("arabuluculuk_6325", 6325),

    # Health / Education / Other
    Statute("milli_egitim_temel_1739", 1739, tertip=5),
    Statute("yuksekogretim_2547", 2547),

    # Constitutional / High Courts
    Statute("aym_kurulusu_6216", 6216),

    # Election / Political
    Statute("siyasi_partiler_2820", 2820),
    Statute("milletvekili_secimi_2839", 2839),

    # Public Procurement / Administrative
    Statute("kamu_ihale_4734", 4734),
    Statute("kamu_ihale_sozlesmeleri_4735", 4735),

    # Foreign / International
    Statute("yabancilar_uluslararasi_koruma_6458", 6458),

    # Forest / Environment
    Statute("orman_6831", 6831, tertip=3),
    Statute("cevre_2872", 2872),
]


def _fetch(url: str, timeout_s: int, insecure: bool) -> bytes:
    """HTTP GET with sensible defaults. Tries requests if available, else urllib."""
    try:
        import requests  # type: ignore

        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_s,
            verify=not insecure,
        )
        resp.raise_for_status()
        return resp.content
    except ImportError:
        pass

    import ssl
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
        return resp.read()


def download_statute(
    s: Statute, out_dir: Path, *, delay_s: float = 1.0, timeout_s: int = 30, insecure: bool = False,
) -> dict:
    """Download one statute as PDF. Returns a manifest record."""
    url = f"{BASE_URL}/{s.tertip}.{s.tur}.{s.no}.pdf"
    out_path = out_dir / f"{s.name}.pdf"
    record = {"name": s.name, "no": s.no, "url": url, "path": str(out_path)}

    if out_path.exists() and out_path.stat().st_size > 1024:
        record["status"] = "already_downloaded"
        record["size_bytes"] = out_path.stat().st_size
        return record

    try:
        content = _fetch(url, timeout_s=timeout_s, insecure=insecure)
        out_path.write_bytes(content)
        record["status"] = "downloaded"
        record["size_bytes"] = len(content)
    except urllib.error.HTTPError as e:
        record["status"] = "http_error"
        record["error_code"] = e.code
        record["error_msg"] = str(e)
    except Exception as e:
        record["status"] = "error"
        record["error_msg"] = str(e)
    finally:
        time.sleep(delay_s)

    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Turkish legal statutes from mevzuat.gov.tr")
    parser.add_argument("--out", required=True, help="Output directory for PDFs")
    parser.add_argument("--delay-s", type=float, default=1.0, help="Polite delay between requests")
    parser.add_argument("--limit", type=int, default=None, help="Only download first N statutes (debug)")
    parser.add_argument("--insecure", action="store_true",
                        help="Skip TLS verification (only for local SSL chain issues; not needed on Colab)")
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = STATUTES[: args.limit] if args.limit else STATUTES
    logger.info("Downloading %d statute(s) to %s", len(targets), out_dir)

    manifest = []
    n_ok = n_skip = n_fail = 0
    for s in targets:
        rec = download_statute(s, out_dir, delay_s=args.delay_s, insecure=args.insecure)
        manifest.append(rec)
        if rec["status"] == "downloaded":
            n_ok += 1
            logger.info("  [%3d] OK     %-50s %d bytes", len(manifest), s.name, rec["size_bytes"])
        elif rec["status"] == "already_downloaded":
            n_skip += 1
            logger.info("  [%3d] SKIP   %-50s already present", len(manifest), s.name)
        else:
            n_fail += 1
            logger.warning("  [%3d] FAIL   %-50s %s", len(manifest), s.name, rec.get("error_msg", rec["status"]))

    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps({
        "downloaded": n_ok,
        "skipped": n_skip,
        "failed": n_fail,
        "total": len(targets),
        "records": manifest,
    }, ensure_ascii=False, indent=2))

    logger.info("Done: %d downloaded, %d skipped, %d failed. Manifest: %s",
                n_ok, n_skip, n_fail, manifest_path)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
