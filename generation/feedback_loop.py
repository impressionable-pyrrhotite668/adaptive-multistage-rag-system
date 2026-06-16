"""Stage 7 — Adaptive Feedback Loop"""
from dataclasses import dataclass
from typing import Optional
from query_router.classifier import RetrievalConfig, QueryType


@dataclass
class FeedbackResult:
    final_answer: object; retries: int; confidence_history: list; improved: bool


class AdaptiveFeedbackLoop:
    def __init__(self, retriever, generator, confidence_threshold=0.45, max_retries=3):
        self.retriever = retriever
        self.generator = generator
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

    def run(self, query, config: RetrievalConfig, filter_doc_id=None):
        best, history = None, []
        for attempt in range(self.max_retries+1):
            cfg = self._adapt(config, attempt)
            q   = self._adapt_query(query, best, attempt)
            print(f"[FeedbackLoop] Attempt {attempt+1} — k={cfg.top_k}")
            ctx = self.retriever.retrieve(q, config=cfg, filter_doc_id=filter_doc_id)
            res = self.generator.generate(ctx)
            history.append(res.confidence)
            print(f"[FeedbackLoop] Confidence: {res.confidence:.3f}")
            if best is None or res.confidence > best.confidence:
                best = res
            if res.confidence >= self.confidence_threshold:
                break
        return FeedbackResult(best, len(history)-1, history,
                              len(history)>1 and history[-1]>history[0])

    def _adapt(self, base, attempt):
        if attempt == 0: return base
        if attempt == 1:
            return RetrievalConfig(base.query_type, min(base.top_k*2,20), True, False,
                                   base.context_budget*2, base.confidence_threshold)
        if attempt == 2:
            return RetrievalConfig(base.query_type, min(base.top_k*3,30), True, False,
                                   int(base.context_budget*2.5), base.confidence_threshold)
        return RetrievalConfig(QueryType.MULTI_HOP, 10, True, True, 4096,
                               base.confidence_threshold)

    @staticmethod
    def _adapt_query(query, prev, attempt):
        if attempt == 0 or prev is None: return query
        existing = set(query.lower().split())
        new = [w.strip(".,;:") for w in prev.answer.split()
               if len(w)>4 and w.isalpha() and w.lower() not in existing][:3]
        return f"{query} (related: {' '.join(new)})" if new else query
