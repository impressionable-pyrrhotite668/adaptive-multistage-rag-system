"""Stage 8 - Evaluation Metrics"""
import re
from dataclasses import dataclass, field

@dataclass
class EvaluationReport:
    query: str; query_type: str
    retrieval_precision: float; context_relevance: float
    answer_faithfulness: float; answer_relevance: float
    chunk_coverage: float; latency_seconds: float
    prompt_tokens: int; completion_tokens: int; total_tokens: int
    passed_threshold: bool; notes: list=field(default_factory=list)
    def summary(self):
        return (f"R-Prec={self.retrieval_precision:.3f} Ctx-Rel={self.context_relevance:.3f} "
                f"Faith={self.answer_faithfulness:.3f} Ans-Rel={self.answer_relevance:.3f} "
                f"Lat={self.latency_seconds:.2f}s Tok={self.total_tokens}")

class RAGEvaluator:
    def __init__(self,use_ragas=False): self.use_ragas=use_ragas
    def evaluate(self,result,context,latency,ground_truth=None):
        notes=[]
        rp=self._rp(context); cr=self._cr(context.query,context.context_text)
        fa=self._fa(result.answer,context.context_text); ar=self._ar(result.answer,context.query)
        cv=self._cv(context)
        u=result.token_usage
        pt=u.get("prompt_tokens",self._et(context.context_text))
        ct=u.get("completion_tokens",self._et(result.answer))
        tt=u.get("total_tokens",pt+ct)
        passed=(rp+cr+fa+ar)/4>=0.5
        if result.confidence<0.4: notes.append(f"Low conf: {result.confidence:.2f}")
        if fa<0.4: notes.append("Potential hallucination")
        return EvaluationReport(query=context.query,query_type=context.query_type,
            retrieval_precision=rp,context_relevance=cr,answer_faithfulness=fa,
            answer_relevance=ar,chunk_coverage=cv,latency_seconds=latency,
            prompt_tokens=pt,completion_tokens=ct,total_tokens=tt,
            passed_threshold=passed,notes=notes)
    def evaluate_batch(self,triples):
        reports=[self.evaluate(r,c,l) for r,c,l in triples]
        m=lambda v:sum(v)/len(v) if v else 0.0
        return {"num_queries":len(reports),
                "avg_retrieval_precision":m([r.retrieval_precision for r in reports]),
                "avg_context_relevance":m([r.context_relevance for r in reports]),
                "avg_faithfulness":m([r.answer_faithfulness for r in reports]),
                "avg_answer_relevance":m([r.answer_relevance for r in reports]),
                "avg_latency":m([r.latency_seconds for r in reports]),
                "pass_rate":m([float(r.passed_threshold) for r in reports]),
                "reports":reports}
    def _rp(self,ctx):
        if not ctx.chunks: return 0.0
        kw=self._kw(ctx.query)
        if not kw: return 0.5
        return sum(1 for c in ctx.chunks if len(kw&set(c.text.lower().split()))>=2)/len(ctx.chunks)
    def _cr(self,query,ctx):
        kw=self._kw(query)
        if not kw: return 0.5
        cl=ctx.lower()
        return sum(1 for w in kw if w in cl)/len(kw)
    def _fa(self,answer,ctx):
        kw=self._kw(answer)
        if not kw: return 0.5
        sc=sum(1 for w in kw if w in ctx.lower())/len(kw)
        for p in ["context does not","not mentioned","insufficient"]:
            if p in answer.lower(): sc=max(sc,0.7)
        return min(1.0,sc)
    def _ar(self,answer,query):
        q,a=self._kw(query),self._kw(answer)
        if not q or not a: return 0.5
        return min(1.0,len(q&a)/len(q))
    def _cv(self,ctx):
        if not ctx.chunks: return 0.0
        return min(1.0,len({c.section_title for c in ctx.chunks if c.section_title})/5.0)
    @staticmethod
    def _kw(text):
        STOP={"the","a","an","in","of","and","or","is","are","was","were","be","been",
              "have","has","had","do","does","did","will","would","to","for","on","at",
              "by","with","this","that","it","what","how","why","when","where","which"}
        return {w for w in re.findall(r"\b\w{3,}\b",text.lower()) if w not in STOP}
    @staticmethod
    def _et(text): return max(1,int(len(text.split())*4/3))
