"""Stage 2 - FAISS/NumPy Vector Store"""
import os, json
import numpy as np
from dataclasses import dataclass

@dataclass
class SearchResult:
    chunk: object; score: float; rank: int
    def __repr__(self): return f"SearchResult(rank={self.rank},score={self.score:.4f})"

class FAISSVectorStore:
    def __init__(self,dim=384,metric="cosine"):
        self.dim=dim; self.metric=metric
        self._chunks={}; self._id_counter=0
        self._np=np.empty((0,dim),dtype="float32")
        self._faiss=self._try()
        self._index=None
    def add_chunks(self,chunks,embs):
        embs=embs.astype("float32")
        assert len(chunks)==len(embs)
        start=self._id_counter
        for i,c in enumerate(chunks): self._chunks[start+i]=c
        self._id_counter+=len(chunks)
        if self._faiss:
            import faiss
            if self._index is None:
                self._index=faiss.IndexFlatIP(self.dim) if self.metric=="cosine" else faiss.IndexFlatL2(self.dim)
            self._index.add(embs)
        else:
            self._np=np.vstack([self._np,embs])
    def search(self,qe,top_k=5,filter_doc_id=None):
        if self._id_counter==0: return []
        q=qe.astype("float32").reshape(1,-1)
        fk=min(top_k*5 if filter_doc_id else top_k, self._id_counter)
        if self._faiss and self._index is not None:
            sc,idx=self._index.search(q,fk)
        else:
            s=(self._np@q.T).flatten()
            ti=np.argsort(s)[::-1][:fk]
            sc=s[ti].reshape(1,-1); idx=ti.reshape(1,-1)
        res=[]
        for s,i in zip(sc[0],idx[0]):
            if i<0: continue
            c=self._chunks.get(int(i))
            if c is None: continue
            if filter_doc_id and c.document_id!=filter_doc_id: continue
            res.append(SearchResult(chunk=c,score=float(s),rank=len(res)+1))
            if len(res)>=top_k: break
        return res
    def get_all_chunks(self): return list(self._chunks.values())
    @property
    def size(self): return self._id_counter
    def save(self,directory):
        os.makedirs(directory,exist_ok=True)
        with open(os.path.join(directory,"chunks.json"),"w") as f:
            json.dump({k:v.to_dict() for k,v in self._chunks.items()},f)
        if self._faiss and self._index is not None:
            import faiss; faiss.write_index(self._index,os.path.join(directory,"faiss.index"))
        else:
            np.save(os.path.join(directory,"np.npy"),self._np)
        with open(os.path.join(directory,"cfg.json"),"w") as f:
            json.dump({"dim":self.dim,"n":self._id_counter,"faiss":self._faiss},f)
        print(f"[Store] Saved {self._id_counter} vectors to {directory!r}")
    def load(self,directory):
        from chunking.semantic_chunker import Chunk
        with open(os.path.join(directory,"cfg.json")) as f: cfg=json.load(f)
        self.dim=cfg["dim"]; self._id_counter=cfg["n"]
        with open(os.path.join(directory,"chunks.json")) as f: raw=json.load(f)
        self._chunks={int(k):Chunk(chunk_id=v["chunk_id"],document_id=v["document_id"],
            text=v["text"],section_title=v.get("section_title",""),
            token_count=v.get("token_count",0),chunk_index=v.get("chunk_index",0),
            start_char=v.get("start_char",0),end_char=v.get("end_char",0),
            metadata=v.get("metadata",{})) for k,v in raw.items()}
        fi=os.path.join(directory,"faiss.index")
        ni=os.path.join(directory,"np.npy")
        if cfg["faiss"] and os.path.exists(fi):
            import faiss; self._index=faiss.read_index(fi)
        elif os.path.exists(ni):
            self._np=np.load(ni)
        print(f"[Store] Loaded {self._id_counter} vectors from {directory!r}")
    @staticmethod
    def _try():
        try: import faiss; return True
        except ImportError: print("[Store] FAISS not found — NumPy fallback."); return False
