"""Cold local smoke test for the evaluator path — runs on a Mac, no GPU, no FAISS.

Simulates what the course evaluator does (reqs 3 & 4): drop in a folder of
documents + a custom Turkish Q&A file, then ingest and benchmark. This exercises
the whole plumbing END-TO-END except generation:

    directory ingest (PDF parse) -> BM25 build -> Turkish {soru, cevap,
    ilgili_belgeler} parsing -> doc-level Recall@k auto-select -> --no-llm
    benchmark -> summary.json / summary.md

It is deliberately BM25-only (``build_faiss=False``) and ``--no-llm`` so it needs
no e5 download and no CUDA — the 4-bit LLM cannot load on a Mac. The full
end-to-end-with-generation check (``--variant prod`` + Qwen-14B) is a separate
Colab dry-run.

Run from the repo root::

    python3 scripts/smoke_evaluator_path.py

Exits 0 and prints ``PASS`` on success; non-zero on any failed assertion so it
can gate CI. Reuses ``src.pipeline.ingest.build_indexes`` and
``src.pipeline.benchmark.run_benchmark`` (no reimplementation).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = REPO_ROOT / "data" / "external" / "mevzuat" / "raw"

# Three real, text-based statute PDFs already in the repo. doc_id after directory
# ingest == the relative filename (flat dir), which is what ``ilgili_belgeler``
# references below — so doc-level recall can match.
PDFS = [
    "ceza_muhakemesi_kanunu_5271.pdf",
    "is_kanunu_4857.pdf",
    "kabahatler_kanunu_5326.pdf",
]

# 6-question Turkish benchmark in the {soru, cevap, ilgili_belgeler} schema an
# evaluator would plausibly use. Questions carry high-signal legal keywords that
# appear verbatim in the target statute, so BM25 surfaces the right document.
# The cevap text is irrelevant here (generation is skipped); only retrieval is scored.
BENCHMARK = [
    {"soru": "Tutuklama kararını hangi merci verir?",
     "cevap": "Tutuklamaya hâkim karar verir.",
     "ilgili_belgeler": ["ceza_muhakemesi_kanunu_5271.pdf"]},
    {"soru": "Gözaltı süresi yakalama anından itibaren ne kadardır?",
     "cevap": "Gözaltı süresi yakalama anından itibaren yirmi dört saattir.",
     "ilgili_belgeler": ["ceza_muhakemesi_kanunu_5271.pdf"]},
    {"soru": "Haftalık çalışma süresi en çok kaç saattir?",
     "cevap": "Haftalık çalışma süresi en çok kırk beş saattir.",
     "ilgili_belgeler": ["is_kanunu_4857.pdf"]},
    {"soru": "İşçinin yıllık ücretli izne hak kazanması için gereken kıdem nedir?",
     "cevap": "İşyerinde en az bir yıl çalışmış olmak gerekir.",
     "ilgili_belgeler": ["is_kanunu_4857.pdf"]},
    {"soru": "Kabahat karşılığında uygulanan idari yaptırımlar nelerdir?",
     "cevap": "İdari para cezası ve idari tedbirlerdir.",
     "ilgili_belgeler": ["kabahatler_kanunu_5326.pdf"]},
    {"soru": "İdari para cezasına karar verme yetkisi kimdedir?",
     "cevap": "Kanunda açıkça gösterilen idari kurul, makam veya kamu görevlilerindedir.",
     "ilgili_belgeler": ["kabahatler_kanunu_5326.pdf"]},
]


def main() -> int:
    from src.generation.rag import RagPipeline
    from src.pipeline.benchmark import run_benchmark
    from src.pipeline.ingest import build_indexes

    missing = [name for name in PDFS if not (PDF_DIR / name).exists()]
    if missing:
        print(f"FAIL: missing test PDFs under {PDF_DIR}: {missing}")
        return 1

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        corpus = td / "corpus"
        corpus.mkdir()
        for name in PDFS:
            shutil.copy(PDF_DIR / name, corpus / name)

        bench_path = td / "bench.jsonl"
        bench_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in BENCHMARK),
            encoding="utf-8",
        )
        idx_dir = td / "indexes"
        report_dir = td / "report"

        # 1. Ingest the folder -> BM25-only bundle (no e5 download).
        manifest = build_indexes(
            inputs=[corpus],
            output_dir=idx_dir,
            build_faiss=False,
            build_bm25=True,
        )
        assert manifest["total_chunks"] > 0, "ingest produced no chunks"
        print(f"  ingest: {manifest['total_chunks']} chunks from {len(PDFS)} PDFs")

        # 2. Load a retrieval-only pipeline (no FAISS, no LLM) and benchmark.
        pipeline = RagPipeline.from_paths(
            index_dir=idx_dir,
            use_faiss=False,
            use_bm25=True,
            load_llm=False,
        )
        summary = run_benchmark(
            pipeline=pipeline,
            benchmark_path=bench_path,
            output_dir=report_dir,
            no_llm=True,
            compute_bertscore=False,
            bootstrap_iterations=0,
            resume=False,
        )

        # 3. Assertions — the core gate.
        assert (report_dir / "summary.json").exists(), "summary.json not written"
        assert (report_dir / "summary.md").exists(), "summary.md not written"
        assert summary.get("generation_skipped") is True, "no-llm mode not honored"
        gran = summary.get("retrieval_granularity")
        assert gran == "doc", f"expected doc-level recall, got granularity={gran!r}"
        ret = summary.get("retrieval") or {}
        recall10 = ret.get("recall@10")
        assert recall10 and recall10 > 0.0, f"doc-level recall@10 is {recall10!r} (expected > 0)"
        # Rank metrics must be in [0,1] — guards the doc-key dedupe (repeated
        # doc-keys would inflate MRR/nDCG above 1.0).
        for k in ("recall@5", "recall@10", "mrr", "ndcg@10"):
            v = ret.get(k)
            assert v is not None and 0.0 <= v <= 1.0, f"{k} out of [0,1]: {v!r}"
        print(f"  retrieval: granularity={gran}  recall@10={recall10:.3f}  "
              f"recall@5={ret.get('recall@5'):.3f}  mrr={ret.get('mrr'):.3f}  ndcg@10={ret.get('ndcg@10'):.3f}")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
