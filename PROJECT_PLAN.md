# CENG493 Turkish Legal RAG - Implementation Plan v3

## Strategy: Solid Core First, Advanced Layers Second

**Weeks 1-4:** Build a complete, demo-ready, polished RAG pipeline (dense + BM25 hybrid, fine-tuned reranker, QLoRA LLM). This alone = top 3.

**Weeks 5-6:** Layer advanced techniques (ColBERT, RAPTOR, KG, Self-RAG, CRAG, agentic orchestration). This = first place.

**Weeks 7-8:** Evaluation, report, presentation.

---

## PHASE 1: CORE PIPELINE (Weeks 1-4) — Demo-Ready by End of Week 4

### WEEK 1 (Mar 18 - Mar 24): Data Engineering + Baseline RAG

**Day 1-2: Corpus Assembly**
- Download all datasets:
  - `erdem-erdem/Turkish-Law-Documents-700k-clustered` (702k Yargitay decisions)
  - `OrionCAF/turkish_law_qa_dataset` (18.3k QA)
  - `Renicames/turkish-law-chatbot` (14.9k QA)
  - Kaggle `turkishlaw-dataset`
- Merge, deduplicate, store as Parquet

**Day 2-3: Legal-Aware Chunking**
- Split on "Madde X", section headers, paragraph breaks
- 512 tokens, 64-token overlap
- Metadata per chunk: law_domain, source, date, court_chamber

**Day 3-4: Gold Test Set**
- 200 standard + 50 adversarial = 250 questions
- 5 domains: Criminal, Civil, Commercial, Administrative, Constitutional
- Adversarial: repealed laws, wrong article numbers, cross-domain, temporal traps
- Format: {question, gold_answer, relevant_doc_ids[], domain, difficulty, is_answerable}

**Day 5-7: Baseline RAG (Ablation Config 1)**
- FAISS IVF-PQ index with `intfloat/multilingual-e5-large` (off-shelf, no FT)
- BM25 index with `rank_bm25` + Turkish stopwords
- RRF fusion (k=60), top-10 to LLM
- `Qwen/Qwen2.5-7B-Instruct` 4-bit, retrieval-aware system prompt
- Implement full evaluation harness:
  - Retrieval: Recall@5, Recall@10, MRR, nDCG@10
  - Generation: EM, F1, ROUGE-L, BLEU
  - Faithfulness + citation accuracy (NLI-based)
- **Run baseline evaluation → record scores**

**Deliverable:** Working E2E pipeline, baseline scores, all data ready.
**Notebook:** `01_data_and_baseline.ipynb`

---

### WEEK 2 (Mar 25 - Mar 31): Embedding Fine-tuning

**Day 1-2: Triplet Generation**
- 50k (query, positive, hard_negative) from QA datasets
- Hard negatives via BM25 (lexically similar, semantically wrong)
- In-batch negatives (same domain, different question)
- Mix 15% English legal triplets (CUAD, ContractNLI) for cross-lingual transfer

**Day 3-5: Fine-tune multilingual-e5-large**
- Loss: MultipleNegativesRankingLoss (sentence-transformers)
- lr=2e-5, batch=32, epochs=3, warmup=0.1
- Evaluate on held-out retrieval set every 500 steps
- Also test from `msbayindir/legal-text-embedding-turkish-v1` (82M, already legal-adapted)

**Day 5-6: Rebuild Indexes + Tune Hybrid**
- Rebuild FAISS with fine-tuned embeddings
- Tune RRF k parameter and BM25/dense weight ratio
- Grid search on validation set

**Day 7: Evaluate (Ablation Config 2)**
- Run full eval with fine-tuned embeddings
- Compare Recall@10 improvement over baseline
- Expect 5-15% Recall@10 gain

**Deliverable:** Fine-tuned embedding, rebuilt indexes, retrieval comparison table.
**Notebook:** `02_embedding_finetune.ipynb`

---

### WEEK 3 (Apr 1 - Apr 7): Reranker Fine-tuning — THE DIFFERENTIATOR

**Day 1-2: Training Data**
- 50k (query, passage, relevance_label) pairs
- 3-level labels: 0=irrelevant, 1=partial, 2=highly relevant
- Include hard negatives from embedding retrieval (passages ranked 5-20 that are wrong)

**Day 3-5: Fine-tune Cross-Encoder**
- Base: `seroe/bge-reranker-v2-m3-turkish-triplet` (already Turkish-tuned)
- lr=1e-6, batch=64, epochs=2, warmup=0.2
- Legal metadata fusion (THE NOVEL CONTRIBUTION):
  ```
  final_score = α·cross_encoder + β·domain_match + γ·recency + δ·court_hierarchy
  ```
  - domain_match: does passage domain match query domain?
  - recency: newer decisions weighted higher
  - court_hierarchy: Yargitay > lower courts
  - Train α/β/γ/δ on validation set

**Day 6-7: Evaluate (Ablation Config 3)**
- Compare: no reranker → off-shelf → fine-tuned → fine-tuned + legal metadata
- Metrics: MRR, nDCG@10, Recall@5 (post-reranking)

**Deliverable:** Fine-tuned reranker with legal features, reranker ablation table.
**Notebook:** `03_reranker_finetune.ipynb`

---

### WEEK 4 (Apr 8 - Apr 14): LLM Fine-tuning + Demo — CORE COMPLETE

**Day 1-2: Instruction Data**
- 15-20k examples from combined QA datasets
- Format:
  ```
  System: Sen bir Turk hukuk uzmanissin. Verilen baglamlara dayanarak soruyu yanitla.
          Her iddiada kaynak goster. Bilgi bulunamadiysa belirt.
  User: Baglam: [passages with doc IDs]
        Soru: [question]
  Assistant: [answer with [Kaynak 1], [Kaynak 2] citations]
  ```
- Include unanswerable examples (teach "bu konuda bilgi bulunamadi")

**Day 3-4: QLoRA Fine-tuning**
- Base: `Qwen/Qwen2.5-7B-Instruct`
- QLoRA: rank=32, alpha=64, dropout=0.05, targets=all linear layers
- lr=3e-5, batch=2, grad_accum=16, epochs=2, seq_len=4096
- Checkpoint to Drive every 500 steps

**Day 5: Evaluate (Ablation Config 4)**
- Run full eval with fine-tuned LLM
- Compare generation quality: base vs fine-tuned

**Day 6-7: Demo System**
- Gradio interface:
  - Input: Turkish legal question
  - Output: answer + citations + passage highlights
  - Show retrieved passages with relevance scores
- Deploy on Colab or HuggingFace Spaces
- Test with 10-15 compelling example questions

**Deliverable:** QLoRA LLM, working Gradio demo, Configs 1-4 ablation complete.
**Notebook:** `04_llm_finetune.ipynb`, `demo.py`

**AT THIS POINT: Core pipeline is complete and demo-ready. Top 3 secured.**

---

## PHASE 2: ADVANCED TECHNIQUES (Weeks 5-6) — Going for First Place

### WEEK 5 (Apr 15 - Apr 21): Advanced Retrieval + Knowledge Graph

**Track A — Contextual Chunking:**
- Run Qwen2.5-7B (4-bit) to generate 1-2 sentence context prefix per chunk
- "Bu belge, 5237 sayili TCK'nin Hayata Karsi Suclar bolumundendir..."
- Re-embed contextualized chunks, add as second FAISS tier
- Batch with checkpointing every 10k chunks

**Track B — Proposition-level Indexing:**
- Decompose chunks into atomic factual propositions via LLM
- Each proposition indexed separately, pointer to parent chunk
- Add as third FAISS tier (raw / contextual / proposition)

**Track C — Knowledge Graph:**
- Parse 702k decisions for legal citations via regex
- Nodes: Laws, Articles, Court Decisions
- Edges: cites, amends, repeals
- NetworkX graph with 2-hop BFS traversal
- Citation chain expansion at retrieval time

**Track D — ColBERT Index (if time):**
- RAGatouille with colbertv2.0
- Fine-tune on same legal triplets from Week 2
- Add as additional retrieval source in RRF fusion

**Deliverable:** Multi-tier index, KG, updated retrieval pipeline.
**Notebook:** `05_advanced_retrieval.ipynb`

---

### WEEK 6 (Apr 22 - Apr 28): Agentic RAG + Self-RAG + CRAG

**Agent Architecture:**

**Agent 1 — Query Intelligence Hub:**
- Domain classifier (fine-tuned BERT, 5 classes)
- Entity extractor (law names, article numbers)
- Query decomposition for complex questions
- HyDE: generate hypothetical answer, embed for retrieval
- Multi-vector: aspect-specific query reformulations

**Agent 2 — Retrieval Orchestrator:**
- Dispatch to all indexes (Dense tiers, BM25, ColBERT if built, KG)
- Per sub-query retrieval for decomposed questions
- RRF fusion + KG citation chain expansion

**Agent 3 — Multi-stage Reranker:**
- Stage 1: passage-level cross-encoder + legal metadata (from Week 3)
- Stage 2: document-level reranking (full doc context)
- CRAG quality gate:
  - Confident (>0.8) → proceed
  - Ambiguous (0.4-0.8) → expand query + re-retrieve
  - Low (<0.4) → flag unanswerable

**Agent 4 — Answer Generator:**
- Self-RAG tokens: [Retrieve], [IsRel], [IsSup], [IsUse]
  - If time: retrain LLM with Self-RAG tokens
  - If not: implement as separate classification step
- Iterative retrieval: after draft, identify gaps, re-retrieve once
- Sub-answer synthesis for decomposed queries

**Agent 5 — Citation Validator:**
- NLI-based claim verification
- Remove/flag unsupported claims
- Calibration MLP for confidence score

**Orchestration:** LangGraph state machine
```
START → QueryAnalysis → RetrievalDispatch → Reranking → CRAGCheck
CRAGCheck ──[confident]──→ Generation → Validation → END
CRAGCheck ──[ambiguous]──→ QueryExpansion → RetrievalDispatch (max 1 loop)
Generation ──[gaps found]──→ IterativeRetrieval → Generation (max 1 loop)
```

**Update Demo:**
- Add debug mode showing each agent's output
- Show KG traversal path, CRAG confidence, Self-RAG decisions

**Deliverable:** Full agentic system (Ablation Configs 5-8), updated demo.
**Notebook:** `06_agentic_rag.ipynb`

---

## PHASE 3: EVALUATION & DELIVERY (Weeks 7-8)

### WEEK 7 (Apr 29 - May 5): Comprehensive Evaluation

**Ablation Study (8 configs):**

| Config | Description |
|--------|-------------|
| C1 | Baseline: off-shelf embedding + BM25 + no reranker + vanilla LLM |
| C2 | + Fine-tuned embeddings (hybrid tuned) |
| C3 | + Fine-tuned reranker + legal metadata |
| C4 | + QLoRA LLM |
| C5 | + Contextual chunks + propositions |
| C6 | + Knowledge Graph + citation chains |
| C7 | + CRAG + document-level reranking |
| C8 | Full agentic system (all techniques) |

**Evaluation Dimensions:**
- Retrieval: Recall@5, Recall@10, MRR, nDCG@10
- Generation: ROUGE-L, BERTScore, F1, EM
- Citation: Precision, Recall
- Calibration: ECE
- Hallucination rate per config

**LLM-as-Judge:**
- GPT-4o + Claude score 250 answers on 5 dimensions:
  factual correctness, completeness, citation accuracy, legal reasoning, hallucination
- Cohen's kappa: GPT-4 vs Claude, each vs human

**Adversarial Robustness (50 questions):**
- Repealed laws, wrong article numbers, cross-domain, temporal traps, fabricated laws
- Per-category accuracy

**Statistical Rigor:**
- Bootstrap 95% CIs (1000 resamples)
- Paired bootstrap + Wilcoxon between consecutive configs
- p-values for every improvement claim

**Hallucination Taxonomy (50 manual annotations):**
- Intrinsic contradiction, extrinsic fabrication, outdated law, jurisdiction confusion, citation fabrication

**Per-Domain Breakdown:**
- All metrics broken down by 5 legal domains
- Radar charts

**Deliverable:** Complete results, all tables and plots.
**Notebook:** `07_evaluation.ipynb`

---

### WEEK 8 (May 6 - May 12): Report, Presentation, Final Demo

**Report (15 pages, LaTeX ACL template):**
1. Introduction & Motivation
2. Related Work
3. System Architecture (full diagram)
4. Methodology (each component)
5. Experiments & Ablation
6. Error Analysis & Hallucination Study
7. Per-domain & Adversarial Analysis
8. Discussion & Limitations
9. Conclusion & Future Work

**Presentation (15 min):**
- Architecture walkthrough
- Live demo: 3-4 compelling examples
- Ablation results with each technique's delta
- Adversarial robustness showcase

**Final Demo (Gradio):**
- Polished UI
- Debug mode with full pipeline visibility
- Hosted on HuggingFace Spaces

**GitHub:**
- Clean README with architecture diagram
- Notebooks 01-07, reproducible
- Model weights on HuggingFace Hub
- Gold + adversarial test sets

---

## Architecture Summary

```
                            USER QUESTION
                                 │
                    ┌────────────▼────────────┐
                    │  AGENT 1: QUERY HUB     │
                    │  • Domain classify       │
                    │  • Entity extract        │
                    │  • Query decompose       │
                    │  • HyDE generate         │
                    │  • Multi-vector embed    │
                    └────────────┬────────────┘
                                │
          ┌─────────────────────▼─────────────────────┐
          │       AGENT 2: RETRIEVAL ORCHESTRATOR      │
          │                                            │
          │  Dense (3-tier)  BM25  ColBERT  KG  RAPTOR │
          │         └────── RRF Fusion ──────┘         │
          │         + KG citation chain expansion      │
          └─────────────────────┬─────────────────────┘
                                │ top-100
          ┌─────────────────────▼─────────────────────┐
          │       AGENT 3: MULTI-STAGE RERANKER        │
          │                                            │
          │  Stage 1: Cross-encoder + legal metadata   │
          │  Stage 2: Document-level reranking         │
          │  CRAG gate → re-retrieve if low confidence │
          └─────────────────────┬─────────────────────┘
                                │ top-10
          ┌─────────────────────▼─────────────────────┐
          │       AGENT 4: ANSWER GENERATOR            │
          │                                            │
          │  Self-RAG QLoRA Qwen2.5-7B                 │
          │  Iterative retrieval for gaps              │
          │  Citation formatting                       │
          └─────────────────────┬─────────────────────┘
                                │
          ┌─────────────────────▼─────────────────────┐
          │       AGENT 5: CITATION VALIDATOR          │
          │                                            │
          │  NLI claim verification                    │
          │  Calibrated confidence score               │
          │  Hallucination flagging                    │
          └─────────────────────┬─────────────────────┘
                                │
                    ANSWER + CITATIONS + CONFIDENCE
```

## Models

| Component | Model | Params |
|-----------|-------|--------|
| Dense Embedding | `intfloat/multilingual-e5-large` | 560M |
| Dense Embedding (alt) | `msbayindir/legal-text-embedding-turkish-v1` | 82M |
| ColBERT | `colbert-ir/colbertv2.0` | 110M |
| Reranker | `seroe/bge-reranker-v2-m3-turkish-triplet` | 560M |
| LLM | `Qwen/Qwen2.5-7B-Instruct` | 7B |
| NLI Validator | `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` | 110M |

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Phase 2 runs behind | Phase 1 is already top-3 quality; cut advanced features gracefully |
| Colab GPU timeout | Checkpoints to Drive every 500 steps |
| No A100 | All FT uses QLoRA/4-bit, fits T4 |
| SPLADE poor Turkish tokenization | Fall back to BM25 (already in Phase 1) |
| Self-RAG retraining too expensive | Implement as separate classification step instead |
| KG too sparse | Even partial graph adds value for adversarial robustness |
