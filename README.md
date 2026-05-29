# Turkish Legal RAG System

A domain-adapted Retrieval-Augmented Generation system for Turkish legal question answering. Hybrid retrieval (FAISS + BM25 + RRF), optional cross-encoder reranking, Qwen2.5-7B generation with optional QLoRA adapter.

## Quickstart (custom corpus + custom benchmark)

The system is designed to be runnable end-to-end on any Turkish legal corpus and any Q&A benchmark — point the CLI at your own data and get scored results back.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Index your corpus (any of: directory of PDFs/.txt/.md, JSONL, JSON array, Parquet).
#    BASE RAG uses the stock encoder; FINE-TUNED RAG uses the fine-tuned E5 checkpoint.
python -m hukuk_rag ingest --docs ./your_corpus/ --out ./indexes/base/
python -m hukuk_rag ingest --docs ./your_corpus/ --out ./indexes/ft/ \
  --embedding-model <path-to-fine-tuned-e5>   # FT-E5 checkpoint (on Google Drive — see Setup)

# 3. BASE RAG vs FINE-TUNED RAG on the SAME benchmark and SAME LLM (the graded comparison)
python -m hukuk_rag benchmark --questions ./your_questions.jsonl \
  --indexes ./indexes/base/ --variant base --report ./reports/base/
python -m hukuk_rag benchmark --questions ./your_questions.jsonl \
  --indexes ./indexes/ft/   --variant prod --report ./reports/ft/

# 4. Ad-hoc query (Fine-tuned RAG)
python -m hukuk_rag query "Kasten adam öldürmenin cezası nedir?" \
  --indexes ./indexes/ft/ --show-passages
```

**Base RAG vs Fine-tuned RAG.** `--variant base` = stock encoder, no reranker, no
snap. `--variant prod` = **Fine-tuned RAG** (fine-tuned E5 index + off-the-shelf
Turkish reranker + source-filtered snap, `repetition_penalty=1.0`, **no QLoRA**) —
this is the recommended, T4/L4-reproducible system that wins our comparison
(+69.1% Token-F1, same 7B LLM). `--variant ft` is the **QLoRA generator**, a
*documented regression* kept only for the ablation — do **not** use it for the
Base-vs-Fine-tuned comparison. Both runs use the same `--llm-base` (default
`Qwen/Qwen2.5-7B-Instruct`; pass `Qwen/Qwen2.5-14B-Instruct` on a ≥40 GB GPU for
the +87.7% scaling result). Add `--no-llm` for GPU-free retrieval-only scoring.

### Accepted input shapes

**Corpus** — auto-detected from path:
- Directory of `.pdf` / `.txt` / `.md` files — chunked via legal-aware chunker (`Madde N` boundaries → sections → paragraphs → token windows).
- JSONL with one of two record schemas:
  - Pre-chunked: `{"id"|"chunk_id": str, "text": str, ...metadata}`
  - Raw documents: `{"id": str, "text": str, ...}` with `text` >2000 chars triggers chunking.
- JSON array (top-level `[...]`) with the same record schema.
- Parquet with `chunk_id` and `text` columns.

**Benchmark** — auto-detected, supports three schemas:
- Our own: `{"questions": [{question, gold_answer, madde_no, kaynak, ...}]}`
- Shared `gold_benchmark.json`: array of `{question_id, question, verified_answer, gold_sources: [{corpus_row_id}]}`
- Shared `rag_eval.json`: array of `{query_id, query, gold_chunk_ids: [str], gold_answer_extract}`
- Any JSONL where each line has `{question|query, answer|verified_answer}` (+ optional `options` for MCQ).

When chunk-level relevance labels are present (`gold_chunk_ids` or `gold_sources`), Recall@K / MRR / nDCG are computed with real ground truth. Otherwise only generation metrics are reported.

## Architecture

```
   QUESTION
      │
      ▼
  ┌───────────────────────────────┐
  │  HYBRID RETRIEVAL             │
  │   Dense (FAISS IVF-PQ)        │
  │   + Sparse (BM25, Turkish)    │
  │   ─── RRF fusion (k=60) ───   │
  └────────────┬──────────────────┘
               │ top-50 each → top-10 after RRF
               ▼
  ┌───────────────────────────────┐
  │  RERANKER (optional)          │
  │   Cross-encoder, BGE-reranker │
  │   -v2-m3-turkish              │
  └────────────┬──────────────────┘
               │ top-10
               ▼
  ┌───────────────────────────────┐
  │  GENERATION                   │
  │   Qwen2.5-7B-Instruct (4-bit) │
  │   + optional QLoRA adapter    │
  │   + Turkish system prompt     │
  └────────────┬──────────────────┘
               │
               ▼
       ANSWER + [Kaynak N] CITATIONS
```

## Key Features

- **Legal-aware chunking** — splits on `Madde N` (article) boundaries, then section headers (BÖLÜM/KISIM/FASIL), then paragraph breaks, falling back to token windows.
- **Turkish language handling** — locale-aware lowercasing (İ→i, I→ı), Turkish stopwords for BM25, legal regex patterns.
- **Fine-tuned embeddings** — `intfloat/multilingual-e5-large` adapted on Turkish legal triplets.
- **Hybrid retrieval** — dense (FAISS IVF-PQ) + sparse (BM25) combined via Reciprocal Rank Fusion (Cormack et al. 2009).
- **Statistical evaluation** — Recall@K, MRR, nDCG@K, Token F1, ROUGE-L, BLEU, faithfulness, citation accuracy. All comparisons reported with bootstrap 95% CIs and Wilcoxon signed-rank significance tests.
- **Cross-LLM gold set audit** — 225-q hand-aligned gold set verified by an independent LLM auditor against canonical mevzuat.gov.tr text. See `reports/gold_audit.json` for full audit verdicts.

## Models

| Component | Model | Parameters |
|-----------|-------|-----------:|
| Embedding | `intfloat/multilingual-e5-large` | 560M |
| Reranker | `seroe/bge-reranker-v2-m3-turkish-triplet` | 560M |
| Generator | `Qwen/Qwen2.5-7B-Instruct` | 7B |
| NLI (faithfulness) | `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` | 110M |

## Datasets

| Dataset | Size | Use |
|---------|------|-----|
| [Turkish-Law-Documents-700k-clustered](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) | 702k decisions | Retrieval corpus |
| [turkish_law_qa_dataset](https://huggingface.co/datasets/OrionCAF/turkish_law_qa_dataset) | 18.3k pairs | Embedding triplets |
| [turkish-law-chatbot](https://huggingface.co/datasets/Renicames/turkish-law-chatbot) | 14.9k pairs | LLM SFT |
| [turkishlaw-dataset](https://www.kaggle.com/datasets/batuhankalem/turkishlaw-dataset-for-llm-finetuning) | 5k pairs | LLM SFT |

## Project Structure

```
├── hukuk_rag/              # CLI entrypoint (python -m hukuk_rag ...)
├── notebooks/              # Experiment notebooks
│   ├── 01_data_and_baseline.ipynb
│   ├── 02_ablation_evaluation.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_phase0_calibration.ipynb
│   └── scratchpads/        # Reference notebooks from prior sessions
├── src/
│   ├── data/               # Chunking, preprocessing, gold set
│   ├── retrieval/          # Dense (FAISS), sparse (BM25), RRF fusion
│   ├── reranker/           # Cross-encoder loader + scoring
│   ├── generation/         # Qwen loader, RAG pipeline, prompts
│   ├── evaluation/         # Metrics, bootstrap CI, Wilcoxon
│   ├── pipeline/           # Ingest, benchmark, CLI
│   └── utils/              # Config, Turkish NLP
├── configs/                # Hyperparameters (config.yaml)
├── data/
│   ├── gold/               # 225-q held-out evaluation set (NEVER trained on)
│   └── external/           # Optional supplementary datasets (gitignored)
├── reports/                # Retrospective, progress report, audit results
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/berkay-aktas/hukuk-rag.git
cd hukuk-rag
pip install -r requirements.txt
```

Designed primarily for **Google Colab with T4 GPU** (16 GB VRAM). QLoRA / 4-bit quantization is used for all LLM operations to fit within memory constraints. Artifacts (indexes, checkpoints, processed corpora) are persisted to Google Drive.

For local use on Mac/Linux without GPU, the ingest and benchmark pipelines work (BM25 is CPU-only, FAISS has CPU support via `faiss-cpu`). LLM generation requires a GPU.

## Ablation Configurations

Six configurations evaluated on the 225-question gold set with a standardized prompt template:

| Config | Embedding | Reranker | Generator |
|--------|-----------|----------|-----------|
| C1 Baseline | Base E5 | — | Base Qwen |
| C2 + FT embedding | FT E5 | — | Base Qwen |
| C3 + FT reranker | FT E5 | FT cross-encoder | Base Qwen |
| C4 + QLoRA LLM | Base E5 | — | QLoRA Qwen |
| C5 FT embed + QLoRA | FT E5 | — | QLoRA Qwen |
| C6 Full system | FT E5 | FT cross-encoder | QLoRA Qwen |

Results, statistical tests, and per-question outputs are in `reports/progress_report.pdf` (April 2026 progress submission). Key findings: embedding fine-tuning yields the largest single-component gain (+19% Token F1, p<0.003); the silver-label-trained reranker degrades downstream quality (−11% F1, p<0.001); QLoRA improves BERTScore (p=0.001) but not Token F1 (p=0.45 — surface-overlap metric undervalues semantic improvement).

## License

This project is for academic research purposes.
