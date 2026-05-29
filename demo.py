"""Gradio demo for the Turkish legal RAG pipeline.

Designed for the course's Colab-with-GPU grading environment: launch with a
public share link, the grader clicks it and interacts with the live system.

Colab (one cell, after the usual bootstrap that mounts Drive + clones the repo)::

    !python demo.py --indexes /content/drive/MyDrive/hukuk-rag/indexes/phase0_shared_ft \
                    --llm-base Qwen/Qwen2.5-7B-Instruct --share

or from inside a notebook cell::

    from demo import build_demo
    build_demo(index_dir="/content/drive/MyDrive/hukuk-rag/indexes/phase0_shared_ft").launch(share=True)

It loads the production configuration via the hardened ``RagPipeline.from_paths``
(off-the-shelf reranker + source-filtered sentence-snap + repetition_penalty=1.0,
no QLoRA). If no CUDA is available it falls back to **retrieval-only** mode so the
UI still opens on a CPU box (answers show the retrieved evidence with a note that
generation needs a GPU) — handy for checking the interface without burning GPU.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = "/content/drive/MyDrive/hukuk-rag/indexes/phase0_shared_ft"
DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"

EXAMPLE_QUESTIONS = [
    "Kasten adam öldürme suçunun cezası nedir?",
    "Tutuklama kararını hangi merci verir?",
    "İş Kanununa göre haftalık çalışma süresi en çok kaç saattir?",
    "Kabahat karşılığında uygulanabilecek idari yaptırımlar nelerdir?",
    "Anayasa Mahkemesine bireysel başvuru süresi kaç gündür?",
]


def _format_passages(results: list[Any], limit: int = 5) -> str:
    """Render retrieved/reranked passages as a Markdown evidence block."""
    if not results:
        return "_Hiç pasaj getirilemedi._"
    lines = []
    for i, r in enumerate(results[:limit], 1):
        md = getattr(r, "metadata", None) or {}
        src = md.get("source") or md.get("source_file") or md.get("doc_id") or md.get("parent_doc_id") or "?"
        text = (r.text or "").strip().replace("\n", " ")
        if len(text) > 500:
            text = text[:500] + "…"
        lines.append(f"**[{i}]** _score={r.score:.3f}_ · `{src}`\n\n> {text}")
    return "\n\n---\n\n".join(lines)


def build_demo(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    llm_base_model: str = DEFAULT_LLM,
    reranker_model: str | None = None,
    use_reranker: bool = True,
    use_faiss: bool = True,
    force_no_llm: bool = False,
):
    """Build the Gradio demo around a loaded :class:`RagPipeline`.

    Args:
        index_dir: Path to an index bundle (built by ``hukuk_rag ingest``).
        llm_base_model: HF id of the base generator.
        reranker_model: Reranker id/path (None → off-the-shelf default).
        use_reranker: Enable the cross-encoder reranker stage.
        force_no_llm: Skip loading the LLM even if CUDA is present (retrieval-only).

    Returns:
        A ``gradio.Blocks`` app (call ``.launch(share=True)`` on it).
    """
    import gradio as gr
    import torch

    from src.generation.rag import RagPipeline

    has_cuda = torch.cuda.is_available()
    load_llm = has_cuda and not force_no_llm
    if not load_llm:
        logger.warning("No CUDA (or --no-llm): running RETRIEVAL-ONLY; answers show evidence only.")

    logger.info("Loading pipeline from %s (load_llm=%s)…", index_dir, load_llm)
    pipeline = RagPipeline.from_paths(
        index_dir=str(index_dir),
        llm_base_model=llm_base_model,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        use_faiss=use_faiss,
        load_llm=load_llm,
        # Production config (mirrors the CLI `--variant prod`).
        repetition_penalty=1.0,
        use_snap_postprocessor=True,
        snap_skip_sources=frozenset({"yargitay"}),
    )

    def answer_fn(question: str, show_evidence: bool):
        question = (question or "").strip()
        if not question:
            return "Lütfen bir soru girin.", ""
        t0 = time.time()
        if load_llm:
            resp = pipeline.answer(question)
            answer = resp.answer or "_(boş yanıt)_"
        else:
            resp = pipeline.retrieve_only(question)
            answer = "⚠️ _Üretim (generation) için GPU gerekir — yalnızca getirilen kanıt gösteriliyor._"
        elapsed = time.time() - t0
        passages = pipeline.final_top_k and (resp.reranked or resp.retrieved)
        evidence = _format_passages(passages or [], limit=5) if show_evidence else ""
        answer = f"{answer}\n\n<sub>⏱ {elapsed:.1f}s · {'üretim+getirme' if load_llm else 'yalnızca getirme'}</sub>"
        return answer, evidence

    mode_note = (
        f"**Model:** `{llm_base_model}` · **Reranker:** {'açık' if use_reranker else 'kapalı'} · "
        f"**Mod:** {'üretim+getirme (GPU)' if load_llm else 'yalnızca getirme (CPU)'}"
    )

    with gr.Blocks(title="Türk Hukuku RAG") as demo:
        gr.Markdown(
            "# ⚖️ Türk Hukuku Soru-Cevap (RAG)\n"
            "Hibrit getirme (yoğun + BM25) → çapraz kodlayıcı yeniden sıralama → "
            "kaynağa dayalı üretim. Yanıtlar getirilen mevzuat/içtihat pasajlarına dayanır.\n\n"
            + mode_note
        )
        with gr.Row():
            q = gr.Textbox(label="Hukuki soru", placeholder="Örn: Kasten adam öldürmenin cezası nedir?", lines=2, scale=4)
        with gr.Row():
            submit = gr.Button("Yanıtla", variant="primary")
            show_ev = gr.Checkbox(label="Kaynak pasajları göster", value=True)
        gr.Markdown("### Yanıt")
        ans = gr.Markdown()
        with gr.Accordion("Getirilen kaynak pasajlar", open=False):
            ev = gr.Markdown()
        gr.Examples(EXAMPLE_QUESTIONS, inputs=q)

        submit.click(answer_fn, inputs=[q, show_ev], outputs=[ans, ev])
        q.submit(answer_fn, inputs=[q, show_ev], outputs=[ans, ev])

    return demo


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Gradio demo for the Turkish legal RAG pipeline.")
    p.add_argument("--indexes", default=DEFAULT_INDEX_DIR, help="Index bundle directory (from `hukuk_rag ingest`).")
    p.add_argument("--llm-base", default=DEFAULT_LLM)
    p.add_argument("--reranker", default=None, help="Reranker id/path (default: off-shelf bge-reranker-turkish).")
    p.add_argument("--no-reranker", action="store_true")
    p.add_argument("--no-faiss", action="store_true", help="Skip dense FAISS (BM25-only) — lets the UI run on a CPU/BM25-only bundle.")
    p.add_argument("--no-llm", action="store_true", help="Retrieval-only (no generation), even if a GPU is present.")
    p.add_argument("--share", action="store_true", help="Create a public gradio.live link (use on Colab).")
    p.add_argument("--server-port", type=int, default=7860)
    args = p.parse_args(argv)

    demo = build_demo(
        index_dir=args.indexes,
        llm_base_model=args.llm_base,
        reranker_model=args.reranker,
        use_reranker=not args.no_reranker,
        use_faiss=not args.no_faiss,
        force_no_llm=args.no_llm,
    )
    demo.launch(share=args.share, server_port=args.server_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
