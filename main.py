"""
main.py — Adaptive Multi-Stage RAG Pipeline Entry Point
========================================================
Usage (run from INSIDE the adaptive_rag folder):

    python main.py --document data/sample.txt --question "What is the main contribution?"
    python main.py --document paper.pdf --question "Summarize this" --backend mock
    python main.py --demo
"""

import argparse, hashlib, json, logging, os, sys, time

# ── CRITICAL FIX: tell Python where to find your modules ─────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# ─────────────────────────────────────────────────────────────────────────────

from ingestion.document_loader    import DocumentLoader
from chunking.semantic_chunker    import SemanticChunker
from embeddings.embedder          import Embedder
from vector_store.faiss_store     import FAISSVectorStore
from retrieval.retriever          import VectorRetriever
from retrieval.reranker           import CrossEncoderReranker
from retrieval.adaptive_retrieval import AdaptiveRetriever
from query_router.classifier      import QueryClassifier
from generation.generator         import LLMGenerator
from generation.feedback_loop     import AdaptiveFeedbackLoop
from evaluation.metrics           import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
#  Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def _index_path(doc_path, store_dir):
    tag = hashlib.md5(os.path.abspath(doc_path).encode()).hexdigest()[:8]
    return os.path.join(store_dir, f"index_{tag}")


def _build_or_load_index(doc_path, store_dir, embed_model, force_rebuild):
    """Load a cached FAISS index or build one from scratch."""
    cache    = _index_path(doc_path, store_dir)
    embedder = Embedder(model_name=embed_model)
    store    = FAISSVectorStore(dim=embedder.embedding_dim)

    config_file = os.path.join(cache, "config.json")
    if not force_rebuild and os.path.exists(config_file):
        logger.info("Loading cached index from %s", cache)
        store.load(cache)
        return store, embedder, store.get_all_chunks()

    logger.info("Building index for '%s' ...", doc_path)
    loader  = DocumentLoader()
    chunker = SemanticChunker(max_chunk_tokens=512)

    if os.path.isdir(doc_path):
        docs = loader.load_directory(doc_path)
    else:
        docs = [loader.load(doc_path)]

    all_chunks = []
    for doc in docs:
        logger.info("  Loaded: '%s' (%d words)", doc.title, doc.word_count)
        chunks = chunker.chunk(doc)
        logger.info("  Chunks: %d", len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No chunks produced — check your document path.")

    logger.info("Embedding %d chunks...", len(all_chunks))
    _, embeddings = embedder.embed_chunks(all_chunks)

    store.add_chunks(all_chunks, embeddings)
    os.makedirs(store_dir, exist_ok=True)
    store.save(cache)
    logger.info("Index saved to %s", cache)

    return store, embedder, all_chunks


def _confidence(answer: str) -> float:
    low = ["don't know", "do not know", "cannot answer",
           "no information", "not mentioned", "context does not"]
    if any(s in answer.lower() for s in low):
        return 0.1
    return min(1.0, len(answer.split()) / 30)


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    doc_path,
    question,
    backend="ollama",
    model="phi3",
    api_base=None,
    store_dir="./index_cache",
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    force_rebuild=False,
    evaluate_flag=False,
    max_feedback_loops=2,
    confidence_threshold=0.3,
    verbose=True,
):
    t0 = time.perf_counter()

    # Stages 1 & 2: load + embed + index
    store, embedder, _ = _build_or_load_index(
        doc_path, store_dir, embed_model, force_rebuild
    )

    # Stage 4: query routing
    clf    = QueryClassifier()
    config = clf.classify(question)
    logger.info("Query type: %s  (k=%d, rerank=%s, iterative=%s)",
                config.query_type, config.top_k, config.rerank, config.iterative)

    # Stages 3.1-3.3: retrieve + rerank + compress
    retriever = VectorRetriever(store, embedder, default_top_k=config.top_k)
    reranker  = CrossEncoderReranker()
    adaptive  = AdaptiveRetriever(retriever, reranker, clf)
    context   = adaptive.retrieve(question, config=config)
    logger.info("Retrieved %d chunks (%d tokens)", len(context.chunks), context.token_count)

    # Stage 6: generation
    generator = LLMGenerator(backend=backend, model_name=model, api_base=api_base)
    result    = generator.generate(context)
    answer    = result.answer

    # Stage 7: feedback loop
    fb_rounds = 0
    while _confidence(answer) < confidence_threshold and fb_rounds < max_feedback_loops:
        fb_rounds += 1
        logger.info("Low confidence — feedback round %d.", fb_rounds)
        from query_router.classifier import RetrievalConfig
        expanded = RetrievalConfig(
            query_type=config.query_type,
            top_k=config.top_k * 2,
            rerank=True,
            iterative=True,
            context_budget=config.context_budget * 2,
            confidence_threshold=config.confidence_threshold,
        )
        context = adaptive.retrieve(question, config=expanded)
        result  = generator.generate(context)
        answer  = result.answer

    latency = time.perf_counter() - t0

    # Stage 8: evaluation
    scores = {}
    if evaluate_flag:
        evaluator = RAGEvaluator()
        report    = evaluator.evaluate(result, context, latency)
        scores = {
            "retrieval_precision": round(report.retrieval_precision, 4),
            "context_relevance":   round(report.context_relevance, 4),
            "answer_faithfulness": round(report.answer_faithfulness, 4),
            "answer_relevance":    round(report.answer_relevance, 4),
            "latency_seconds":     round(latency, 3),
            "total_tokens":        report.total_tokens,
        }

    output = {
        "question":        question,
        "answer":          answer,
        "query_type":      config.query_type,
        "chunks_used":     len(context.chunks),
        "backend":         result.model_used,
        "feedback_rounds": fb_rounds,
        "latency_seconds": round(latency, 3),
        "scores":          scores,
    }

    if verbose:
        _print_result(output, context.chunks)

    return output


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(result, chunks):
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  Question    : {result['question']}")
    print(f"  Query type  : {result['query_type']}")
    print(f"  Backend     : {result['backend']}")
    print(f"  Chunks used : {result['chunks_used']}")
    print(f"  Latency     : {result['latency_seconds']} s")
    print(sep)
    print(f"\n  ANSWER\n  {'-'*40}")
    for line in result["answer"].split("\n"):
        print(f"  {line}")
    print()
    if result["scores"]:
        print(f"  EVALUATION SCORES\n  {'-'*40}")
        for k, v in result["scores"].items():
            if isinstance(v, float):
                print(f"  {k:<28} {v:.4f}")
            else:
                print(f"  {k:<28} {v}")
        print()
    print(f"  TOP RETRIEVED CHUNKS\n  {'-'*40}")
    for i, chunk in enumerate(chunks[:3], 1):
        title   = chunk.section_title or "No section"
        snippet = chunk.text[:200].replace("\n", " ")
        print(f"  [{i}] Section: {title[:50]}")
        print(f"      {snippet}...")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Demo mode
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    demo_path = os.path.join(HERE, "data", "demo_document.txt")

    if not os.path.exists(demo_path):
        from data.sample_generator import generate_sample_document
        os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
        generate_sample_document(demo_path)
        print(f"[Demo] Created: {demo_path}")

    questions = [
        "What is the main contribution of this research?",
        "Summarize the methodology used in this study.",
        "Why did the authors choose this approach over alternatives?",
        "How do the results relate to the introduction hypothesis?",
    ]

    print("\n" + "="*70)
    print("  ADAPTIVE RAG -- DEMO MODE")
    print("="*70)

    for q in questions:
        print(f"\n>>> {q}")
        run_pipeline(
            doc_path=demo_path,
            question=q,
            backend="mock",
            evaluate_flag=True,
            verbose=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adaptive Multi-Stage RAG Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--document",    default="")
    parser.add_argument("--question",    default="")
    parser.add_argument("--backend",     default="ollama",
                        choices=["mock", "openai", "huggingface", "ollama", "stub", "auto"])
    parser.add_argument("--model",       default="phi3")
    parser.add_argument("--api-base",    default=None)
    parser.add_argument("--embed-model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--store-dir",   default="./index_cache")
    parser.add_argument("--rebuild",     action="store_true")
    parser.add_argument("--evaluate",    action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--demo",        action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        sys.exit(0)

    if not args.document or not args.question:
        parser.error("--document and --question are required  (or use --demo)")

    result = run_pipeline(
        doc_path      = args.document,
        question      = args.question,
        backend       = args.backend,
        model         = args.model,
        api_base      = args.api_base,
        store_dir     = args.store_dir,
        embed_model   = args.embed_model,
        force_rebuild = args.rebuild,
        evaluate_flag = args.evaluate,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Saved to {args.output_json}]")