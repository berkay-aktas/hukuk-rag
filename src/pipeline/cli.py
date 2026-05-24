"""CLI entrypoint: ``python -m hukuk_rag {ingest,benchmark,query}``.

Designed for the course evaluator workflow. Two commands cover Requirements 3
and 4 (custom corpus + custom benchmark):

::

    # 1. Build indexes from any corpus format
    python -m hukuk_rag ingest --docs ./my_corpus_dir/ --out ./indexes/
    python -m hukuk_rag ingest --docs corpus.jsonl --out ./indexes/

    # 2. Score a custom Q&A file against those indexes
    python -m hukuk_rag benchmark \\
        --questions ./my_questions.jsonl \\
        --indexes ./indexes/ \\
        --variant ft \\
        --report ./report/

    # 3. Ad-hoc query
    python -m hukuk_rag query "Kasten öldürmenin cezası nedir?" --indexes ./indexes/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Dispatch to ``ingest``, ``benchmark``, or ``query`` subcommand.

    Returns:
        Process exit code.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        prog="hukuk_rag",
        description="Turkish legal RAG pipeline — ingest, benchmark, query.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Build FAISS + BM25 indexes from a corpus.")
    p_ingest.add_argument("--docs", required=True, action="append",
                          help="Path to corpus (jsonl/json/parquet/directory). Can be repeated.")
    p_ingest.add_argument("--out", required=True, help="Output directory for the index bundle.")
    p_ingest.add_argument("--embedding-model", default="intfloat/multilingual-e5-large",
                          help="HF model id or local path. Default: multilingual-e5-large.")
    p_ingest.add_argument("--no-faiss", action="store_true", help="Skip FAISS build.")
    p_ingest.add_argument("--no-bm25", action="store_true", help="Skip BM25 build.")
    p_ingest.add_argument("--chunk-max-tokens", type=int, default=480)
    p_ingest.add_argument("--chunk-overlap-tokens", type=int, default=64)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Score a Q&A file against built indexes.")
    p_bench.add_argument("--questions", required=True, help="Path to benchmark file (json/jsonl).")
    p_bench.add_argument("--indexes", required=True, help="Path to index bundle directory.")
    p_bench.add_argument("--report", required=True, help="Output report directory.")
    p_bench.add_argument("--variant", choices=["base", "ft", "ft+rerank"], default="ft",
                         help="Which model variant. base=vanilla Qwen, ft=QLoRA, ft+rerank=QLoRA+cross-encoder.")
    p_bench.add_argument("--qlora-adapter", help="Path to QLoRA adapter dir (required for ft/ft+rerank).")
    p_bench.add_argument("--llm-base", default="Qwen/Qwen2.5-7B-Instruct")
    p_bench.add_argument("--reranker", help="Reranker model id or path (default: off-shelf bge-reranker-turkish).")
    p_bench.add_argument("--prompt", default="default",
                         choices=["default", "short", "strict_citation", "mcq"])
    p_bench.add_argument("--bootstrap-n", type=int, default=1000)

    # query
    p_query = subparsers.add_parser("query", help="Run one ad-hoc question.")
    p_query.add_argument("question", help="The question text (Turkish).")
    p_query.add_argument("--indexes", required=True, help="Path to index bundle directory.")
    p_query.add_argument("--qlora-adapter", help="Optional QLoRA adapter path.")
    p_query.add_argument("--use-reranker", action="store_true")
    p_query.add_argument("--show-passages", action="store_true",
                         help="Print retrieved passages alongside the answer.")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "query":
        return _cmd_query(args)
    parser.print_help()
    return 1


def _cmd_ingest(args) -> int:
    from src.pipeline.ingest import build_indexes

    manifest = build_indexes(
        inputs=args.docs,
        output_dir=args.out,
        embedding_model=args.embedding_model,
        build_faiss=not args.no_faiss,
        build_bm25=not args.no_bm25,
        chunker_max_tokens=args.chunk_max_tokens,
        chunker_overlap_tokens=args.chunk_overlap_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _cmd_benchmark(args) -> int:
    from src.generation.prompts import (
        CITATION_STRICT_SYSTEM, DEFAULT_SYSTEM, MCQ_SYSTEM, SHORT_ANSWER_SYSTEM,
    )
    from src.generation.rag import RagPipeline
    from src.pipeline.benchmark import run_benchmark

    prompt_map = {
        "default": DEFAULT_SYSTEM,
        "short": SHORT_ANSWER_SYSTEM,
        "strict_citation": CITATION_STRICT_SYSTEM,
        "mcq": MCQ_SYSTEM,
    }

    use_reranker = args.variant.endswith("+rerank")
    qlora = args.qlora_adapter if args.variant != "base" else None
    if args.variant != "base" and not args.qlora_adapter:
        print("WARNING: --variant ft/ft+rerank set but --qlora-adapter not provided — using base Qwen.",
              file=sys.stderr)

    pipeline = RagPipeline.from_paths(
        index_dir=args.indexes,
        llm_base_model=args.llm_base,
        qlora_adapter=qlora,
        reranker_model=args.reranker,
        use_reranker=use_reranker,
        system_prompt=prompt_map[args.prompt],
    )

    summary = run_benchmark(
        pipeline=pipeline,
        benchmark_path=args.questions,
        output_dir=args.report,
        bootstrap_iterations=args.bootstrap_n,
    )
    print(f"\nDone. Report: {Path(args.report).resolve()}")
    print(f"  Token F1: {summary['generation']['token_f1']:.4f} "
          f"[95% CI {summary['token_f1_95ci']['ci_low']:.4f}, {summary['token_f1_95ci']['ci_high']:.4f}]")
    if summary.get("retrieval"):
        print(f"  Recall@10: {summary['retrieval']['recall@10']:.4f}, MRR: {summary['retrieval']['mrr']:.4f}")
    return 0


def _cmd_query(args) -> int:
    from src.generation.rag import RagPipeline

    pipeline = RagPipeline.from_paths(
        index_dir=args.indexes,
        qlora_adapter=args.qlora_adapter,
        use_reranker=args.use_reranker,
    )
    resp = pipeline.answer(args.question)
    print("\n" + "=" * 72)
    print(f"SORU: {args.question}")
    print("=" * 72)
    print(f"\nCEVAP:\n{resp.answer}\n")
    if args.show_passages:
        print("-" * 72)
        print("RETRIEVED PASSAGES:")
        for i, r in enumerate(resp.reranked or resp.retrieved[:5], 1):
            print(f"\n[{i}] score={r.score:.4f}  chunk_id={r.chunk_id}")
            print(r.text[:300] + ("..." if len(r.text) > 300 else ""))
    print(f"\nTiming: {resp.timing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
