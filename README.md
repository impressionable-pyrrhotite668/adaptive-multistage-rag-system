# Adaptive Multi-Stage RAG System

A production-ready, research-level **Adaptive Multi-Stage Retrieval-Augmented Generation** pipeline for long documents.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE RAG PIPELINE                        │
│                                                                     │
│  Document  ──►  Ingestion  ──►  Chunking  ──►  Embeddings  ──►  FAISS │
│                                                                     │
│  Query  ──►  Router  ──►  Stage 3.1 Vector Search                  │
│                      ──►  Stage 3.2 Cross-Encoder Re-Rank           │
│                      ──►  Stage 3.3 Context Compression             │
│                                                                     │
│                 ──►  LLM Generation  ──►  Answer                   │
│                                                                     │
│                 ──►  Feedback Loop (if low confidence)             │
│                                                                     │
│                 ──►  Evaluation Metrics                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
adaptive_rag/
├── ingestion/
│   └── document_loader.py      # PDF / DOCX / TXT loading + cleaning
├── chunking/
│   └── semantic_chunker.py     # Adaptive semantic chunking
├── embeddings/
│   └── embedder.py             # Sentence-Transformers batched encoding
├── vector_store/
│   └── faiss_store.py          # FAISS IndexFlatIP + metadata store
├── retrieval/
│   ├── retriever.py            # Stage 3.1 – vector retrieval
│   ├── reranker.py             # Stage 3.2 – cross-encoder re-ranking
│   └── adaptive_retrieval.py  # Stage 3.3 – dedup + merge + budget
├── query_router/
│   └── classifier.py          # Stage 4 – query type classification
├── generation/
│   └── generator.py           # Stage 6 – LLM generation (OpenAI / HF / stub)
├── evaluation/
│   └── metrics.py             # Stage 8 – evaluation metrics + RAGAS
├── data/
│   └── sample.txt             # Example document (Transformer paper overview)
├── tests/
│   └── test_pipeline.py       # Full test suite (unit + integration)
├── main.py                    # CLI entry point
└── requirements.txt
```

---

## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK sentence tokeniser
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Optional — OpenAI API

```bash
export OPENAI_API_KEY="sk-..."
```

### Optional — HuggingFace local model

```bash
pip install transformers accelerate
export HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
```

---

## Quick Start

```bash
# Answer a question about the bundled sample document (stub backend — no API key needed)
python -m adaptive_rag.main \
    --document adaptive_rag/data/sample.txt \
    --question "What is the main contribution of the Transformer?" \
    --backend stub \
    --evaluate

# Use OpenAI (requires OPENAI_API_KEY)
python -m adaptive_rag.main \
    --document my_paper.pdf \
    --question "Summarise the methodology" \
    --backend openai \
    --model gpt-4o \
    --evaluate

# Save JSON output
python -m adaptive_rag.main \
    --document report.pdf \
    --question "What are the key findings?" \
    --output-json result.json
```

---

## CLI Reference

| Flag            | Default            | Description                                          |
|-----------------|--------------------|------------------------------------------------------|
| `--document`    | *(required)*       | Path to PDF / TXT / DOCX                            |
| `--question`    | *(required)*       | Query to answer                                      |
| `--backend`     | `auto`             | `auto` / `openai` / `huggingface` / `stub`          |
| `--model`       | `gpt-3.5-turbo`    | LLM model name                                       |
| `--embed-model` | `all-MiniLM-L6-v2` | Sentence-Transformers model                          |
| `--store-dir`   | `./index_cache`    | FAISS index cache directory                          |
| `--rebuild`     | `false`            | Force re-index even if cache exists                  |
| `--evaluate`    | `false`            | Compute evaluation metrics                           |
| `--output-json` | `""`               | Write result dict to JSON file                       |

---

## Pipeline Stages

### Stage 1 — Document Ingestion
`ingestion/document_loader.py`

Supports PDF (PyMuPDF → pdfplumber fallback), DOCX, TXT/Markdown.
Extracts per-page text, cleans Unicode, guesses section titles.

### Stage 1b — Adaptive Chunking
`chunking/semantic_chunker.py`

1. Splits on paragraph / double-newline boundaries.
2. Merges blocks smaller than `min_tokens`.
3. Splits blocks larger than `max_tokens` at sentence boundaries with configurable overlap.

### Stage 2 — Embeddings + Vector Store
`embeddings/embedder.py` · `vector_store/faiss_store.py`

Batched encoding via `sentence-transformers` (default `all-MiniLM-L6-v2`).
L2-normalised vectors stored in FAISS `IndexFlatIP` → exact cosine search.
Index is serialised to disk; subsequent runs load from cache.

### Stage 3 — Multi-Stage Retrieval

| Sub-stage   | File                          | What it does                                      |
|-------------|-------------------------------|---------------------------------------------------|
| **3.1**     | `retrieval/retriever.py`      | Top-k cosine similarity search                   |
| **3.2**     | `retrieval/reranker.py`       | Cross-encoder re-ranking (ms-marco MiniLM)        |
| **3.3**     | `retrieval/adaptive_retrieval.py` | Dedup · merge · token-budget enforcement     |

### Stage 4 — Query Routing
`query_router/classifier.py`

Rule-based regex classifier assigns one of:
`factual` · `summarization` · `reasoning` · `multi_hop`

Each type maps to a different `RetrievalConfig` (k, token budget, iterative flag).

### Stage 5 — Long Context Handling
Handled within chunking (adaptive chunk sizes) and the compression step
(token budget, deduplication, section-aware merging).

### Stage 6 — LLM Generation
`generation/generator.py`

Builds a structured prompt `[context + question + instructions]` and calls:
- **OpenAI** (any chat-completions compatible endpoint)
- **HuggingFace** local model via `transformers.pipeline`
- **Stub** (returns the prompt — no model required, for testing)

### Stage 7 — Adaptive Feedback Loop
`main.py` (`run_pipeline`)

If the generated answer is short / contains "I don't know", the pipeline
automatically expands retrieval (2× k) and regenerates (up to `max_feedback_loops` times).

### Stage 8 — Evaluation Metrics
`evaluation/metrics.py`

| Metric                | Method                                              |
|-----------------------|-----------------------------------------------------|
| `retrieval_precision` | Token-Jaccard overlap between query and chunks      |
| `context_relevance`   | Mean cosine similarity of chunk embeddings to query |
| `answer_faithfulness` | Fraction of answer sentences supported by context  |
| `answer_relevance`    | Cosine similarity of answer embedding to query      |
| `token_usage`         | Approximate total tokens consumed                   |
| `latency_seconds`     | Wall-clock time for full pipeline                   |
| RAGAS (optional)      | `faithfulness` + `answer_relevancy` via RAGAS lib   |

---

## Running Tests

```bash
pip install pytest
python -m pytest adaptive_rag/tests/ -v
```

---

## Performance Optimisation Suggestions

| Concern          | Recommendation                                                          |
|------------------|-------------------------------------------------------------------------|
| Index speed      | Switch to `IndexIVFFlat` with nprobe tuning for > 100k chunks          |
| Embedding speed  | Use `all-mpnet-base-v2` for quality or `all-MiniLM-L12-v2` for speed   |
| Re-ranking speed | Use `cross-encoder/ms-marco-TinyBERT-L-2` for 4× faster re-ranking     |
| Memory           | Use `IndexFlatIP` + memmap for corpora that don't fit in RAM            |
| Latency          | Cache query embeddings; pre-compute chunk embeddings once              |
| Long docs        | Enable hierarchical retrieval: embed section summaries separately      |
| Accuracy         | Fine-tune the cross-encoder on domain-specific relevance labels         |

---

## Example Output

```
──────────────────────────────────────────────────────────────────────
  Question   : What is the main contribution of the Transformer?
  Query type : factual
  Backend    : openai
  Chunks used: 3
  Latency    : 1.842 s
──────────────────────────────────────────────────────────────────────

  ANSWER
  ────────────────────────────────────────
  The Transformer's main contribution is demonstrating that attention
  alone is sufficient for high-quality sequence modelling, removing
  the need for recurrence. This enables parallelism and scaling to
  billions of parameters.

  EVALUATION SCORES
  ────────────────────────────────────────
  retrieval_precision          0.8333
  context_relevance            0.7241
  answer_faithfulness          0.8750
  answer_relevance             0.8102
  token_usage                  487
  latency_seconds              1.842

  RETRIEVED CHUNKS (top-3 shown)
  ────────────────────────────────────────
  [1] 9. Conclusion  (score=8.421)
      The Transformer's main contribution is demonstrating that attention
      alone is sufficient for high-quality sequence modelling…
──────────────────────────────────────────────────────────────────────
```

---

## License

MIT — free for academic and research use.
