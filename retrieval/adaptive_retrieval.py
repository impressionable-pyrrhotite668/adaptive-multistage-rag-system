"""Stage 3.3 + 4 + 5 - Adaptive Retrieval Pipeline"""
import re, uuid, numpy as np
from dataclasses import dataclass, field

@dataclass
class RetrievedContext:
    query: str; query_type: str; chunks: list; context_text: str; token_count: int
    retrieval_steps: int=1; metadata: dict=field(default_factory=dict)

class AdaptiveRetriever:
    def __init__(self,retriever,reranker,classifier,max_hops=3):
        self.retriever=retriever; self.reranker=reranker
        self.classifier=classifier; self.max_hops=max_hops
    def retrieve(self,query,config=None,filter_doc_id=None):
        if config is None: config=self.classifier.classify(query)
        if config.iterative: return self._mhop(query,config,filter_doc_id)
        return self._single(query,config,filter_doc_id)
    def _single(self,query,config,fid):
        from query_router.classifier import QueryType
        mode="hierarchical" if config.query_type==QueryType.SUMMARIZATION else "flat"
        res=self.retriever.retrieve(query,top_k=config.top_k,mode=mode,filter_doc_id=fid)
        if config.rerank and res: res=self.reranker.rerank(query,res)
        chunks=self._compress([r.chunk for r in res],query,config.context_budget)
        ctx=self._asm(chunks)
        return RetrievedContext(query=query,query_type=config.query_type,chunks=chunks,
                                context_text=ctx,token_count=self._tok(ctx),retrieval_steps=1)
    def _mhop(self,query,config,fid):
        all_c,seen,sq,steps=[],set(),query,0
        for _ in range(self.max_hops):
            steps+=1
            res=self.retriever.retrieve(sq,top_k=config.top_k,filter_doc_id=fid)
            if config.rerank and res: res=self.reranker.rerank(sq,res)
            new=[r.chunk for r in res if r.chunk.chunk_id not in seen]
            for c in new: seen.add(c.chunk_id)
            all_c.extend(new)
            if sum(c.token_count for c in all_c)>=config.context_budget: break
            fu=self._fu(query,all_c)
            if not fu or fu==sq: break
            sq=fu
        comp=self._compress(all_c,query,config.context_budget)
        ctx=self._asm(comp)
        return RetrievedContext(query=query,query_type=config.query_type,chunks=comp,
                                context_text=ctx,token_count=self._tok(ctx),retrieval_steps=steps)
    def _compress(self,chunks,query,budget):
        if not chunks: return chunks
        chunks=self._dedup(chunks)
        kw=self._kw(query)
        if kw: chunks=[c for c in chunks if len(kw&set(c.text.lower().split()))>0] or chunks
        chunks=self._merge(chunks,budget)
        return self._trim(chunks,budget)
    def _dedup(self,chunks,t=0.85):
        kept,ks=[],[]
        for c in chunks:
            toks=set(c.text.lower().split())
            if not any(len(toks&e)/max(len(toks|e),1)>=t for e in ks):
                kept.append(c); ks.append(toks)
        return kept
    def _merge(self,chunks,budget):
        if len(chunks)<=1: return chunks
        from chunking.semantic_chunker import Chunk
        merged=[chunks[0]]
        for c in chunks[1:]:
            last=merged[-1]
            if (last.section_title==c.section_title and last.document_id==c.document_id
                and abs(c.chunk_index-last.chunk_index)<=2
                and last.token_count+c.token_count<=budget//2):
                merged[-1]=Chunk(chunk_id=str(uuid.uuid4())[:8],document_id=last.document_id,
                                 text=last.text+" "+c.text,section_title=last.section_title,
                                 token_count=last.token_count+c.token_count,
                                 chunk_index=last.chunk_index,
                                 start_char=last.start_char,end_char=c.end_char,
                                 metadata=last.metadata)
            else: merged.append(c)
        return merged
    def _trim(self,chunks,budget):
        kept,used=[],0
        for c in chunks:
            if used+c.token_count>budget: break
            kept.append(c); used+=c.token_count
        return kept or chunks[:1]
    @staticmethod
    def _asm(chunks):
        return "\n---\n".join(f"[Section: {c.section_title}]\n{c.text}" if c.section_title else f"[Document]\n{c.text}" for c in chunks)
    @staticmethod
    def _fu(query,chunks):
        if not chunks: return query
        orig=set(query.lower().split())
        ctx=" ".join(c.text for c in chunks[:3])
        caps=re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",ctx)
        new=[t for t in caps if t.lower() not in orig]
        return f"{query} specifically regarding {new[0]}" if new else query
    @staticmethod
    def _kw(text):
        STOP={"the","a","an","in","of","and","or","is","are","was","were","be","been",
              "have","has","had","do","does","did","will","would","to","for","on","at","by",
              "with","this","that","it","what","how","why","when","where","which","who"}
        return {w for w in re.findall(r"\b\w{3,}\b",text.lower()) if w not in STOP}
    @staticmethod
    def _tok(text): return max(1,int(len(text.split())*4/3))
