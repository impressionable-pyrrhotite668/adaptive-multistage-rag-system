"""
data/sample_generator.py
=========================
Generates a realistic synthetic research paper for demo and testing purposes.
No external dependencies required.
"""

SAMPLE_PAPER = """
# Adaptive Retrieval-Augmented Generation for Long Document Question Answering

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding, yet they struggle with long-document question answering due to context window limitations. We present AdaptiveRAG, a multi-stage retrieval-augmented generation framework that dynamically adjusts its retrieval strategy based on query type classification. Our system employs a three-stage retrieval pipeline: initial vector search, cross-encoder re-ranking, and context compression, combined with an adaptive feedback loop that expands the search when answer confidence is low. Experimental results on standard benchmarks demonstrate that AdaptiveRAG achieves a 23% improvement in answer faithfulness and a 31% reduction in hallucination rate compared to naive RAG baselines. The framework supports documents of arbitrary length through hierarchical chunking and section-aware retrieval.

## 1. Introduction

The proliferation of large language models has transformed natural language processing. However, a fundamental challenge remains: LLMs have fixed context windows, typically ranging from 4,096 to 128,000 tokens, while real-world documents—such as legal contracts, scientific papers, and technical manuals—often exceed these limits. The naive approach of truncating documents leads to significant information loss.

Retrieval-Augmented Generation (RAG) addresses this limitation by retrieving relevant document segments at query time and including them in the LLM's context. However, vanilla RAG systems treat all queries uniformly, applying the same retrieval strategy regardless of whether the question requires a simple factual lookup or a complex multi-hop reasoning chain spanning multiple document sections.

Our key insight is that query complexity and type fundamentally determine the optimal retrieval strategy. A question asking "What year was this study conducted?" requires only a single precise chunk, while "How does the proposed method compare to state-of-the-art alternatives across different experimental settings?" requires broad coverage of multiple sections and iterative retrieval.

The main contributions of this paper are:
1. A query type classifier that categorises queries into factual, summarization, reasoning, and multi-hop types.
2. A multi-stage retrieval pipeline with cross-encoder re-ranking and context compression.
3. An adaptive feedback loop that detects low-confidence answers and expands retrieval.
4. Comprehensive evaluation demonstrating significant improvements over baseline RAG systems.

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

The concept of augmenting language model generation with retrieved information was introduced by Lewis et al. (2020), who demonstrated that retrieval could significantly improve open-domain question answering. Subsequent work has explored various aspects of RAG systems, including better retrieval mechanisms, improved chunking strategies, and more sophisticated prompting approaches.

Dense Passage Retrieval (DPR) established the use of bi-encoders for passage retrieval, enabling fast approximate nearest-neighbour search. The ColBERT model improved upon this by using late interaction between query and passage representations, achieving better retrieval quality at manageable computational cost.

### 2.2 Re-Ranking in Information Retrieval

Cross-encoders have long been used as re-rankers in two-stage retrieval pipelines. Unlike bi-encoders that encode queries and documents independently, cross-encoders jointly process (query, document) pairs, enabling richer interaction modelling. MS-MARCO benchmarks have established cross-encoders as the gold standard for passage re-ranking.

### 2.3 Long Document Processing

Hierarchical approaches to long document processing have been explored in the summarization literature. Work on multi-document summarization demonstrated that section-aware processing outperforms flat chunking strategies. More recently, sparse attention mechanisms have been proposed to extend transformer context windows, though these approaches require model fine-tuning.

## 3. Methodology

### 3.1 Document Ingestion and Chunking

Our ingestion pipeline processes PDF, TXT, and DOCX files using PyMuPDF and python-docx respectively. Raw text undergoes cleaning to remove artefacts common in PDF extraction: hyphenated line breaks, page numbers, headers, and encoding errors.

The semantic chunker identifies section boundaries using a combination of heuristic rules and pattern matching. Heading patterns include Markdown-style headers, numbered section markers, and ALL-CAPS section titles. Within sections, text is split at paragraph boundaries and merged until reaching a configurable token budget (default: 512 tokens). When a single paragraph exceeds the token budget, sentence-level splitting is applied.

Chunk overlap—repeating the last sentence from the previous chunk at the start of the next—ensures that information spanning chunk boundaries is not lost. Each chunk carries metadata including document ID, section title, character offsets, and token count.

### 3.2 Embedding Generation

We use the all-MiniLM-L6-v2 sentence transformer model, which produces 384-dimensional embeddings. This model offers an excellent trade-off between semantic quality and computational efficiency, processing approximately 14,000 sentences per second on a modern CPU. All embeddings are L2-normalised to enable cosine similarity computation via inner products.

Embeddings are stored in a FAISS IndexFlatIP index, which supports exact inner-product search. For larger deployments, the index can be replaced with an IVFPQ index for approximate search with sub-linear query time.

### 3.3 Multi-Stage Retrieval

The retrieval pipeline consists of three stages:

**Stage 1 — Vector Retrieval**: The user query is embedded using the same sentence transformer model. A top-k nearest-neighbour search is performed against the FAISS index. The value of k is determined by the query type classifier.

**Stage 2 — Cross-Encoder Re-Ranking**: Retrieved candidates are re-scored using the cross-encoder/ms-marco-MiniLM-L-6-v2 model. This joint query-passage scoring significantly improves relevance ranking. Re-ranking is applied selectively: factual queries skip this stage for efficiency, while reasoning and multi-hop queries always re-rank.

**Stage 3 — Context Compression**: The re-ranked chunks undergo a four-step compression pipeline:
1. Deduplication using Jaccard similarity (threshold: 0.85)
2. Relevance filtering using query keyword overlap
3. Adjacent chunk merging for same-section chunks
4. Token budget trimming to fit within the LLM context window

### 3.4 Adaptive Query Routing

The query classifier uses a two-level approach. The first level applies pattern matching using curated regular expressions for each query type. Factual queries are characterised by WH-words (what, who, when, where) combined with specific entity mentions. Summarization queries contain terms like "summarize", "overview", or "main points". Reasoning queries feature causal language (why, how does, explain, analyse). Multi-hop queries contain relational terms (both, throughout, connection between, relationship).

The second level, optional for production use, applies zero-shot classification using a BART-large-MNLI model for ambiguous queries. This provides approximately 8% improvement in classification accuracy at the cost of additional latency.

### 3.5 Adaptive Feedback Loop

When the generated answer scores below a confidence threshold (default: 0.45), the feedback loop initiates a retry with expanded retrieval. Three escalation levels are defined:

- **Level 1**: Double the retrieval k, enable re-ranking
- **Level 2**: Triple k, expand context budget
- **Level 3**: Switch to full multi-hop iterative retrieval

Confidence is estimated heuristically using answer length, presence of uncertainty phrases, and keyword overlap between the answer and the retrieved context. While this is a proxy metric, it correctly identifies low-quality answers in 78% of cases in our validation experiments.

## 4. Experiments

### 4.1 Experimental Setup

We evaluate AdaptiveRAG on three datasets:
- **NaturalQuestions** (NQ): Open-domain factual questions derived from Google Search queries
- **QuALITY**: Long-document reading comprehension with questions requiring multi-sentence evidence
- **SummEval**: Document summarization quality evaluation

All experiments use the all-MiniLM-L6-v2 embedding model and GPT-3.5-Turbo as the generation backbone. Baselines include naive RAG (fixed k=5, no re-ranking), BM25 retrieval, and HyDE (Hypothetical Document Embedding).

### 4.2 Results

On the NaturalQuestions benchmark, AdaptiveRAG achieves an exact match score of 41.3%, compared to 35.7% for naive RAG and 38.2% for BM25. The improvement is most pronounced on multi-hop questions (27% relative improvement), where iterative retrieval captures evidence distributed across multiple sections.

Answer faithfulness, measured using GPT-4 as a judge, improves by 23% over the naive RAG baseline. This improvement is primarily attributed to the context compression stage, which removes noisy and irrelevant chunks that can confuse the generation model.

Latency benchmarks show that the cross-encoder re-ranking stage adds approximately 180ms on CPU for 10 candidates. The adaptive feedback loop incurs no additional latency when the initial answer has sufficient confidence (73% of queries in our test set).

### 4.3 Ablation Study

We conduct ablation studies by removing individual pipeline components:
- Removing cross-encoder re-ranking: -12% faithfulness
- Removing context compression: -8% faithfulness, +45% token usage
- Disabling adaptive routing (fixed k=5): -18% on multi-hop, +7% on factual
- Disabling feedback loop: -6% overall faithfulness

The results confirm that each component contributes meaningfully to overall performance.

## 5. Discussion

### 5.1 Failure Modes

The system struggles on queries that require precise numerical reasoning (e.g., "By what percentage did X increase compared to Y in Table 3?"). The sentence transformer embeddings are not optimised for numerical content, leading to poor retrieval of passages containing tables and figures. Future work should explore specialised encoders for structured content.

Another failure mode occurs with very short documents (< 500 words), where the chunking strategy produces only a few chunks that may not differentiate well by query type. The adaptive routing provides limited benefit in these cases.

### 5.2 Computational Considerations

The full pipeline (embedding + retrieval + re-ranking + generation) completes in 2.3 seconds on average using CPU inference for the embedding and re-ranking models, plus API latency for generation. Inference time scales linearly with document length during indexing but is independent of document length at query time—a key advantage of RAG over long-context LLMs.

## 6. Conclusion

We have presented AdaptiveRAG, a multi-stage retrieval-augmented generation system that adapts its retrieval strategy to the complexity and type of the user's query. Through query classification, cross-encoder re-ranking, context compression, and an adaptive feedback loop, the system achieves significant improvements over baseline RAG approaches while maintaining reasonable latency.

The modular architecture allows individual components to be replaced or upgraded independently. Future directions include incorporating fine-tuned embedding models for domain-specific corpora, neural rerankers trained on domain-specific relevance judgments, and integration with structured knowledge bases for hybrid retrieval.

## References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
2. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
3. Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction. SIGIR.
4. Gao, L., et al. (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels. ACL.
5. Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain QA. EACL.
6. Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv.
"""


def generate_sample_document(output_path: str = "demo_document.txt") -> str:
    """Write the sample research paper to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_PAPER.strip())
    return output_path


if __name__ == "__main__":
    path = generate_sample_document()
    print(f"Sample document written to: {path}")
    print(f"Word count: {len(SAMPLE_PAPER.split())}")
