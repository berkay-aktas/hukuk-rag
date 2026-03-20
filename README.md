# Turkish Legal RAG System

A domain-adapted Retrieval-Augmented Generation system for Turkish legal question answering, built on Yargıtay (Supreme Court) decisions and statutory law.

## Architecture

```
                         USER QUESTION
                              │
                 ┌────────────▼────────────┐
                 │     QUERY ANALYSIS       │
                 │  • Domain classification │
                 │  • Entity extraction     │
                 │  • Query decomposition   │
                 └────────────┬────────────┘
                              │
       ┌──────────────────────▼──────────────────────┐
       │          HYBRID RETRIEVAL                    │
       │                                             │
       │  Dense (FAISS IVF-PQ)  ·  Sparse (BM25)    │
       │  Fine-tuned E5-large   ·  Turkish tokenizer │
       │           └──── RRF Fusion ────┘            │
       └──────────────────────┬──────────────────────┘
                              │ top-k
       ┌──────────────────────▼──────────────────────┐
       │          CROSS-ENCODER RERANKER              │
       │                                             │
       │  Fine-tuned BGE-reranker-v2-m3-turkish      │
       │  + Legal metadata fusion (domain, recency,  │
       │    court hierarchy)                          │
       └──────────────────────┬──────────────────────┘
                              │ top-10
       ┌──────────────────────▼──────────────────────┐
       │          ANSWER GENERATION                   │
       │                                             │
       │  QLoRA fine-tuned Qwen2.5-7B-Instruct       │
       │  + Citation formatting                      │
       │  + NLI-based claim verification             │
       └──────────────────────┬──────────────────────┘
                              │
                  ANSWER + CITATIONS + CONFIDENCE
```

## Key Features

- **Legal-aware chunking** — splits on `Madde` (article) boundaries, section headers, and paragraph breaks before falling back to token-based splitting
- **Turkish language handling** — proper locale lowercasing (İ→i, I→ı), Turkish stopword lists, legal-specific regex patterns
- **Fine-tuned embeddings** — `intfloat/multilingual-e5-large` adapted on 61k Turkish legal triplets with hard negatives
- **Hybrid retrieval** — dense (FAISS IVF-PQ) + sparse (BM25) combined via Reciprocal Rank Fusion
- **Evaluation suite** — Recall@K, MRR, nDCG, token F1, ROUGE-L with bootstrap 95% CIs and Wilcoxon significance tests

## Models

| Component | Model | Parameters |
|-----------|-------|------------|
| Embedding | `intfloat/multilingual-e5-large` | 560M |
| Embedding (alt) | `msbayindir/legal-text-embedding-turkish-v1` | 82M |
| ColBERT | `colbert-ir/colbertv2.0` | 110M |
| Reranker | `seroe/bge-reranker-v2-m3-turkish-triplet` | 560M |
| Generator | `Qwen/Qwen2.5-7B-Instruct` | 7B |
| NLI | `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` | 110M |

## Datasets

| Dataset | Size | Use |
|---------|------|-----|
| [Turkish-Law-Documents-700k-clustered](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) | 702k decisions | RAG corpus |
| [turkish_law_qa_dataset](https://huggingface.co/datasets/OrionCAF/turkish_law_qa_dataset) | 18.3k pairs | QA training |
| [turkish-law-chatbot](https://huggingface.co/datasets/Renicames/turkish-law-chatbot) | 14.9k pairs | QA training |
| [turkishlaw-dataset](https://www.kaggle.com/datasets/batuhankalem/turkishlaw-dataset-for-llm-finetuning) | — | Fine-tuning |

## Project Structure

```
├── notebooks/              # Experiment notebooks (01-07)
│   └── 01_data_and_baseline.ipynb
├── src/
│   ├── data/               # Chunking, preprocessing, gold set management
│   ├── retrieval/          # Dense (FAISS), sparse (BM25), RRF fusion
│   ├── evaluation/         # Metrics, bootstrap CI, significance tests
│   └── utils/              # Config, Turkish NLP, device management
├── configs/
│   └── config.yaml         # Hyperparameters, model identifiers, paths
├── data/
│   └── gold/               # Held-out evaluation set (never trained on)
└── requirements.txt
```

## Setup

```bash
# Clone
git clone https://github.com/berkay-aktas/hukuk-rag.git
cd hukuk-rag

# Install dependencies
pip install -r requirements.txt
```

The system is designed for **Google Colab with T4 GPU** (16 GB VRAM). All fine-tuning uses QLoRA/4-bit quantization to fit within memory constraints. Artifacts (indexes, checkpoints, processed data) are persisted to Google Drive.

## Usage

Each notebook is self-contained and runnable top-to-bottom on a fresh Colab session. Outputs are checkpointed to Drive, so interrupted sessions resume automatically.

```python
from src.retrieval import dense_search, bm25_search, rrf_merge, load_faiss_index, load_bm25_index
from src.evaluation import retrieval_metrics, generation_metrics, bootstrap_ci
```

## Ablation Configurations

| Config | Description |
|--------|-------------|
| C1 | Baseline: off-shelf E5 + BM25 + RRF + vanilla Qwen 4-bit |
| C2 | + Fine-tuned embeddings |
| C3 | + Fine-tuned reranker + legal metadata fusion |
| C4 | + QLoRA fine-tuned LLM |
| C5 | + Contextual chunks + proposition indexing |
| C6 | + Knowledge graph + citation chains |
| C7 | + CRAG + document-level reranking |
| C8 | Full agentic system |

## License

This project is for academic research purposes.
