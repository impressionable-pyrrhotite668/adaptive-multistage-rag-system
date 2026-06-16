"""Stage 3.1 + Stage 5 - Vector Retrieval"""
import numpy as np
from typing import Optional
from vector_store.faiss_store import FAISSVectorStore, SearchResult

class VectorRetriever:
    def __init__(self, vector_store, embedder, default_top_k=5):
        self.vector_store=vector_store; self.embedder=embedder; self.default_top_k=default_top_k
    def retrieve(self, query, top_k=None, mode="flat", filter_doc_id=None):
        k=top_k or self.default_top_k
        qe=self.embedder.embed_query(query)
        if mode=="hierarchical": return self._hier(qe,k,filter_doc_id)
        elif mode=="section": return self._sec(qe,k,filter_doc_id)
        return self.vector_store.search(qe,top_k=k,filter_doc_id=filter_doc_id)
    def _hier(self,qe,k,fid):
        cands=self.vector_store.search(qe,top_k=k*3,filter_doc_id=fid)
        if not cands: return []
        secs={}
        for r in cands: secs.setdefault(r.chunk.section_title or "__",[] ).append(r)
        ss={s:np.mean([r.score for r in rs]) for s,rs in secs.items()}
        final,seen=[],set()
        for s in sorted(ss,key=ss.get,reverse=True):
            for r in sorted(secs[s],key=lambda x:x.score,reverse=True):
                if r.chunk.chunk_id not in seen: final.append(r); seen.add(r.chunk.chunk_id)
                if len(final)>=k: break
            if len(final)>=k: break
        for i,r in enumerate(final,1): r.rank=i
        return final
    def _sec(self,qe,q,k,fid):
        all_c=self.vector_store.get_all_chunks()
        secs={}
        for c in all_c:
            if fid and c.document_id!=fid: continue
            secs.setdefault(c.section_title or "__",[]).append(c)
        if not secs: return self.vector_store.search(qe,top_k=k,filter_doc_id=fid)
        titles=list(secs.keys())
        te=self.embedder.embed_texts(titles)
        bi=int(np.argmax((te@qe).flatten()))
        sc=secs[titles[bi]]
        ce=self.embedder.embed_texts([c.text for c in sc])
        cs=(ce@qe).flatten()
        ti=np.argsort(cs)[::-1][:k]
        return [SearchResult(chunk=sc[i],score=float(cs[i]),rank=r+1) for r,i in enumerate(ti)]
