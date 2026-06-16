# Adaptive Multi-Stage RAG System

[![CI](https://github.com/Geeta3521/adaptive-multistage-rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Geeta3521/adaptive-multistage-rag-system/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready, research-level **Adaptive Multi-Stage Retrieval-Augmented Generation (RAG)** pipeline for long-document question answering. Supports PDF, DOCX, and TXT documents with multi-stage retrieval, cross-encoder re-ranking, adaptive feedback loops, and a full evaluation suite.

📄 [Read the research paper](./final_paper_after_check.pdf)

---

## Why this project?

Most open-source RAG systems stop at vector search. This pipeline goes further:

- **3-stage retrieval** — vector search → cross-encoder re-ranking → context compression
- **Query-aware routing** — retrieval strategy adapts to whether the query is factual, summarization, reasoning, or multi-hop
- **Adaptive feedback loop** — low-confidence answers automatically trigger expanded retrieval and regeneration
- **Full evaluation** — retrieval precision, context relevance, answer faithfulness, and answer relevance out of the box

---

## Architecture

```
Document ──► Ingestion ──► Chunking ──► Embeddings ──► FAISS Index
                                                              │
Query ──► Query Router ──► Stage 3.1  Vector Search          │◄──────┐
                       ──► Stage 3.2  Cross-Encoder Re-Rank         │
                       ──► Stage 3.3  Context Compression            │
                                 │                                   │
                       ──► LLM Generation ──► Answer                 │
                                 │                                   │
                       ──► Confidence Check ──► Feedback Loop ───────┘
                                 │
                       ──► Evaluation Metrics
```

---

## Project Structure

```
adaptive-multistage-rag-system/
├── ingestion/
│   └── document_loader.py       # PDF / DOCX / TXT loading + cleaning
├── chunking/
│   └── semantic_chunker.py      # Adaptive semantic chunking
├── embeddings/
│   └── embedder.py              # Sentence-Transformers batched encoding
├── vector_store/
│   └── faiss_store.py           # FAISS IndexFlatIP + metadata store
├── retrieval/
│   ├── retriever.py             # Stage 3.1 – vector retrieval
│   ├── reranker.py              # Stage 3.2 – cross-encoder re-ranking
│   └── adaptive_retrieval.py    # Stage 3.3 – dedup + merge + token budget
├── query_router/
│   └── classifier.py            # Query type classification
├── generation/
│   ├── generator.py             # LLM generation (OpenAI / HuggingFace / stub)
│   └── feedback_loop.py         # Adaptive feedback loop
├── evaluation/
│   └── metrics.py               # Evaluation metrics + optional RAGAS
├── scripts/
│   ├── create_notebook.py
│   ├── sample_generator.py
│   └── run_eval.py
├── .github/workflows/
│   └── ci.yml                   # GitHub Actions CI
├── main.py                      # CLI entry point
├── pyproject.toml
├── requirements.txt
├── research_notebook.ipynb
├── adaptive_rag_evaluation.ipynb
└── eval_results.json
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/Geeta3521/adaptive-multistage-rag-system.git
cd adaptive-multistage-rag-system

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install the package
pip install -e .

# 4. Download NLTK sentence tokeniser
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Optional — OpenAI backend

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
# Run with the built-in demo (no API key needed)
python main.py --demo

# Answer a question using the stub backend (no API key needed)
python main.py \
  --document data/sample.txt \
  --question "What is the main contribution of the Transformer?" \
  --backend stub \
  --evaluate

# Use OpenAI
python main.py \
  --document my_paper.pdf \
  --question "Summarise the methodology" \
  --backend openai \
  --model gpt-4o \
  --evaluate

# Save output as JSON
python main.py \
  --document report.pdf \
  --question "What are the key findings?" \
  --output-json result.json
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--document` | *(required)* | Path to PDF / TXT / DOCX |
| `--question` | *(required)* | Query to answer |
| `--backend` | `auto` | `auto` / `openai` / `huggingface` / `ollama` / `stub` |
| `--model` | `gpt-3.5-turbo` | LLM model name |
| `--embed-model` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `--store-dir` | `./index_cache` | FAISS index cache directory |
| `--rebuild` | `false` | Force re-index even if cache exists |
| `--evaluate` | `false` | Compute evaluation metrics |
| `--output-json` | `""` | Write result dict to a JSON file |
| `--demo` | `false` | Run the built-in demo (no document needed) |

---

## Pipeline Stages

### Stage 1 — Document ingestion
`ingestion/document_loader.py`

Supports PDF (PyMuPDF with pdfplumber fallback), DOCX, and TXT/Markdown. Extracts per-page text, cleans Unicode, and infers section titles.

### Stage 1b — Adaptive chunking
`chunking/semantic_chunker.py`

Splits on paragraph boundaries, merges blocks below `min_tokens`, and splits blocks above `max_tokens` at sentence boundaries with configurable overlap. Produces clean, semantically coherent chunks rather than fixed-size windows.

### Stage 2 — Embeddings and vector store
`embeddings/embedder.py` · `vector_store/faiss_store.py`

Batched encoding via `sentence-transformers` (default `all-MiniLM-L6-v2`). L2-normalised vectors stored in FAISS `IndexFlatIP` for exact cosine search. The index is serialised to disk so subsequent runs load from cache instantly.

### Stage 3 — Multi-stage retrieval

| Sub-stage | File | What it does |
|---|---|---|
| 3.1 | `retrieval/retriever.py` | Top-k cosine similarity search |
| 3.2 | `retrieval/reranker.py` | Cross-encoder re-ranking (`ms-marco-MiniLM`) |
| 3.3 | `retrieval/adaptive_retrieval.py` | Dedup · merge · token-budget enforcement |

### Stage 4 — Query routing
`query_router/classifier.py`

Classifies each query as `factual`, `summarization`, `reasoning`, or `multi_hop`. Each type maps to a different `RetrievalConfig` controlling top-k, token budget, re-ranking, and whether iterative retrieval is used.

### Stage 5 — Long context handling
Handled by adaptive chunk sizes (Stage 1b) and the token-budget compression step (Stage 3.3). No separate truncation step — context fits within the LLM window by construction.

### Stage 6 — LLM generation
`generation/generator.py`

Builds a structured prompt `[context + question + instructions]` and calls:
- **OpenAI** — any chat-completions compatible endpoint
- **HuggingFace** — local model via `transformers.pipeline`
- **Ollama** — local models via the Ollama API
- **Stub** — returns the prompt directly, no model required (great for testing)

### Stage 7 — Adaptive feedback loop
`generation/feedback_loop.py` · `main.py`

If the generated answer has low confidence (measured by cosine similarity between the answer embedding and the context), the pipeline automatically doubles the retrieval budget and regenerates — up to `max_feedback_loops` times.

### Stage 8 — Evaluation
`evaluation/metrics.py`

| Metric | Method |
|---|---|
| `retrieval_precision` | Token-Jaccard overlap between query and chunks |
| `context_relevance` | Mean cosine similarity of chunk embeddings to query |
| `answer_faithfulness` | Fraction of answer sentences supported by context |
| `answer_relevance` | Cosine similarity of answer embedding to query |
| `token_usage` | Approximate total tokens consumed |
| `latency_seconds` | Wall-clock time for the full pipeline |
| RAGAS *(optional)* | `faithfulness` + `answer_relevancy` via the RAGAS library |

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
  ──────────────────────────────────────
  The Transformer's main contribution is demonstrating that attention
  alone is sufficient for high-quality sequence modelling, removing
  the need for recurrence. This enables parallelism and scaling to
  billions of parameters.

  EVALUATION SCORES
  ──────────────────────────────────────
  retrieval_precision          0.8333
  context_relevance            0.7241
  answer_faithfulness          0.8750
  answer_relevance             0.8102
  token_usage                  487
  latency_seconds              1.842
──────────────────────────────────────────────────────────────────────
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

---

## Performance Tuning

| Concern | Recommendation |
|---|---|
| Index speed | Switch to `IndexIVFFlat` with nprobe tuning for > 100k chunks |
| Embedding quality | Use `all-mpnet-base-v2` for higher accuracy |
| Embedding speed | Use `all-MiniLM-L12-v2` for faster encoding |
| Re-ranking speed | Use `cross-encoder/ms-marco-TinyBERT-L-2` for 4× faster re-ranking |
| Memory | Use `IndexFlatIP` + memmap for corpora that don't fit in RAM |
| Latency | Cache query embeddings; pre-compute chunk embeddings once |
| Long docs | Enable hierarchical retrieval — embed section summaries separately |
| Accuracy | Fine-tune the cross-encoder on domain-specific relevance labels |

---

## Roadmap

- [ ] Streaming generation support
- [ ] REST API wrapper (FastAPI)
- [ ] Streamlit / Gradio demo UI
- [ ] Support for web URLs as document sources
- [ ] Hierarchical retrieval for very large document sets
- [ ] Docker container for one-command deployment

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change. Make sure all tests pass before submitting a pull request.

```bash
pip install -e ".[dev]"
pytest -v
```

---

## License

MIT — free for academic and commercial use. See [LICENSE](LICENSE) for details.

---

## Author

**Geetha** · [GitHub](https://github.com/Geeta3521)

If you find this project useful, please consider giving it a ⭐ on GitHub!
