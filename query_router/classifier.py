"""Stage 4 - Query Type Classifier"""
import re
from dataclasses import dataclass

class QueryType:
    FACTUAL="factual"; SUMMARIZATION="summarization"
    REASONING="reasoning"; MULTI_HOP="multi_hop"

@dataclass
class RetrievalConfig:
    query_type:str; top_k:int; rerank:bool; iterative:bool
    context_budget:int; confidence_threshold:float
    def __str__(self): return f"Config(type={self.query_type},k={self.top_k})"

_C={
    "factual":        RetrievalConfig("factual",3,False,False,512,0.5),
    "summarization":  RetrievalConfig("summarization",10,True,False,2048,0.4),
    "reasoning":      RetrievalConfig("reasoning",7,True,False,1536,0.45),
    "multi_hop":      RetrievalConfig("multi_hop",5,True,True,2048,0.5),
}

class QueryClassifier:
    _P={
        "factual":      [r"what is",r"who is",r"when (was|did|is)",r"where (is|was)",r"define",r"how (many|much|old|long)"],
        "summarization":[r"summari[sz]e",r"overview",r"briefly",r"main (idea|point|finding)s?",r"tldr",r"outline"],
        "reasoning":    [r"why",r"how does",r"explain",r"analyze",r"analyse",r"compare",r"evaluate",r"discuss"],
        "multi_hop":    [r"both",r"throughout",r"across",r"relationship between",r"how .+ relate",r"connection between"],
    }
    def __init__(self):
        self._cp={qt:[re.compile(p,re.I) for p in ps] for qt,ps in self._P.items()}
    def classify(self,query):
        sc={qt:sum(1 for p in ps if p.search(query)) for qt,ps in self._cp.items()}
        priority=["multi_hop","reasoning","summarization","factual"]
        best=max(priority,key=lambda qt:sc[qt])
        qt=best if sc[best]>0 else "reasoning"
        cfg=_C[qt]; print(f"[Router] '{query[:50]}' -> {qt} (k={cfg.top_k})")
        return cfg
    def get_config(self,qt): return _C.get(qt,_C["factual"])
