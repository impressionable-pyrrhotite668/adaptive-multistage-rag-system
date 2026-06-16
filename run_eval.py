import sys
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r"D:\adaptive_rag_fixed")

from main import _build_or_load_index
from query_router.classifier import QueryClassifier, RetrievalConfig
from retrieval.retriever import VectorRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.adaptive_retrieval import AdaptiveRetriever
from generation.generator import LLMGenerator
from evaluation.metrics import RAGEvaluator

print("Starting evaluation runner...")

DOC_PATH = "final_paper_after_check.pdf"
STORE_DIR = "./index_cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading index...")
store, embedder, all_chunks = _build_or_load_index(DOC_PATH, STORE_DIR, EMBED_MODEL, force_rebuild=False)

clf = QueryClassifier()
retriever = VectorRetriever(store, embedder, default_top_k=3)
reranker = CrossEncoderReranker()
adaptive_retriever = AdaptiveRetriever(retriever, reranker, clf)
evaluator = RAGEvaluator()

backend_type = "ollama"
llm_model = "qwen2.5:1.5b"
generator = LLMGenerator(backend=backend_type, model_name=llm_model)

test_queries = [
    "What is the main contribution of this paper regarding air quality prediction?",
    "How does the GA-Opt RF model compare against the SVR model in terms of RMSE?",
    "What dataset was used for evaluating the synthetic results?"
]

results_data = []

print("Running queries...")
for i, query in enumerate(test_queries):
    print(f"\n--- Query {i+1} ---")
    
    # Baseline
    t0 = time.perf_counter()
    baseline_config = RetrievalConfig(query_type="factual", top_k=3, rerank=False, iterative=False, context_budget=512, confidence_threshold=0.5)
    ctx_baseline = adaptive_retriever.retrieve(query, config=baseline_config)
    res_baseline = generator.generate(ctx_baseline)
    lat_baseline = time.perf_counter() - t0
    eval_baseline = evaluator.evaluate(res_baseline, ctx_baseline, lat_baseline)
    
    results_data.append({
        "Query": f"Q{i+1}",
        "Method": "Baseline RAG",
        "Retrieval Precision": eval_baseline.retrieval_precision,
        "Context Relevance": eval_baseline.context_relevance,
        "Answer Faithfulness": eval_baseline.answer_faithfulness,
        "Answer Relevance": eval_baseline.answer_relevance,
        "Latency (s)": eval_baseline.latency_seconds
    })
    
    # Adaptive
    t0 = time.perf_counter()
    adaptive_config = clf.classify(query)
    ctx_adaptive = adaptive_retriever.retrieve(query, config=adaptive_config)
    res_adaptive = generator.generate(ctx_adaptive)
    lat_adaptive = time.perf_counter() - t0
    eval_adaptive = evaluator.evaluate(res_adaptive, ctx_adaptive, lat_adaptive)
    
    results_data.append({
        "Query": f"Q{i+1}",
        "Method": "Adaptive RAG",
        "Retrieval Precision": eval_adaptive.retrieval_precision,
        "Context Relevance": eval_adaptive.context_relevance,
        "Answer Faithfulness": eval_adaptive.answer_faithfulness,
        "Answer Relevance": eval_adaptive.answer_relevance,
        "Latency (s)": eval_adaptive.latency_seconds
    })

print("Writing results to eval_results.json...")
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=4)
print("Complete.")
