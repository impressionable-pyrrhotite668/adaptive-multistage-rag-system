"""Stage 1 (continued) - Semantic Chunking"""
import re, uuid
from dataclasses import dataclass, field

@dataclass
class Chunk:
    chunk_id: str; document_id: str; text: str
    section_title: str = ""; token_count: int = 0
    chunk_index: int = 0; start_char: int = 0; end_char: int = 0
    metadata: dict = field(default_factory=dict)
    def to_dict(self):
        return {"chunk_id":self.chunk_id,"document_id":self.document_id,"text":self.text,
                "section_title":self.section_title,"token_count":self.token_count,
                "chunk_index":self.chunk_index,"start_char":self.start_char,
                "end_char":self.end_char,"metadata":self.metadata}

class SemanticChunker:
    _HP = [re.compile(p) for p in [
        r"^#{1,6}\s+.+$",
        r"^(\d+\.)+\d*\s+[A-Z].+$",
        r"^[A-Z][A-Z\s]{4,}$",
        r"^(Abstract|Introduction|Conclusion|Background|Method|Results|Discussion|References|Appendix)\b",
    ]]
    def __init__(self,max_chunk_tokens=512,min_chunk_tokens=64,overlap_sentences=1):
        self.max_chunk_tokens=max_chunk_tokens
        self.min_chunk_tokens=min_chunk_tokens
        self.overlap_sentences=overlap_sentences
        self._ss=None
    def chunk(self, doc):
        sections=self._parse(doc.text)
        chunks,idx,pos=[],0,0
        for title,paras in sections:
            nc=self._chunk_sec(paras,doc.document_id,title,pos,idx)
            for c in nc: c.metadata.update({"document_title":doc.title,"file_path":doc.file_path})
            chunks.extend(nc); idx+=len(nc)
            pos+=sum(len(p) for p in paras)+len(title)+4
        return chunks
    def _parse(self,text):
        lines=text.split("\n"); sections=[]; h=""; buf=[]
        for l in lines:
            if self._head(l):
                if buf:
                    p=self._l2p(buf)
                    if p: sections.append((h,p))
                h=l.strip("# ").strip(); buf=[]
            else: buf.append(l)
        if buf:
            p=self._l2p(buf)
            if p: sections.append((h,p))
        if not sections:
            p=self._l2p(lines)
            sections=[("",p)]
        return sections
    def _head(self,line):
        s=line.strip()
        if not s or len(s)>120: return False
        return any(p.match(s) for p in self._HP)
    @staticmethod
    def _l2p(lines):
        ps,buf=[],[]
        for l in lines:
            if l.strip(): buf.append(l.strip())
            else:
                if buf: ps.append(" ".join(buf)); buf=[]
        if buf: ps.append(" ".join(buf))
        return [p for p in ps if p.strip()]
    def _chunk_sec(self,paras,did,title,pos,si):
        chunks,buf,bt=[],[],0; idx=si
        for para in paras:
            pt=self._tok(para)
            if pt>self.max_chunk_tokens:
                if buf: chunks.append(self._mk(did,title," ".join(buf),idx,pos)); pos+=len(" ".join(buf)); idx+=1; buf,bt=[],0
                for sc in self._split_s(para,did,title,idx,pos): chunks.append(sc); pos+=len(sc.text); idx+=1
            elif bt+pt>self.max_chunk_tokens:
                if buf: chunks.append(self._mk(did,title," ".join(buf),idx,pos)); pos+=len(" ".join(buf)); idx+=1
                ov=self._ov(buf); buf=ov+[para]; bt=sum(self._tok(t) for t in buf)
            else: buf.append(para); bt+=pt
        if buf: chunks.append(self._mk(did,title," ".join(buf),idx,pos))
        return chunks
    def _split_s(self,text,did,title,si,pos):
        sents=self._ss_split(text); chunks,buf,bt=[],[],0; idx=si
        for s in sents:
            st=self._tok(s)
            if bt+st>self.max_chunk_tokens and buf:
                chunks.append(self._mk(did,title," ".join(buf),idx,pos)); pos+=len(" ".join(buf)); idx+=1
                ov=self._ov(buf); buf=ov+[s]; bt=sum(self._tok(t) for t in buf)
            else: buf.append(s); bt+=st
        if buf: chunks.append(self._mk(did,title," ".join(buf),idx,pos))
        return chunks
    def _mk(self,did,title,text,idx,start):
        text=text.strip()
        return Chunk(chunk_id=str(uuid.uuid4())[:8],document_id=did,text=text,
                     section_title=title,token_count=self._tok(text),
                     chunk_index=idx,start_char=start,end_char=start+len(text))
    def _ov(self,buf):
        if not self.overlap_sentences: return []
        all_s=[]
        for t in buf: all_s.extend(self._ss_split(t))
        return all_s[-self.overlap_sentences:]
    @staticmethod
    def _tok(text): return max(1,int(len(text.split())*4/3))
    def _ss_split(self,text):
        if self._ss is None:
            try:
                import nltk
                try: nltk.data.find("tokenizers/punkt_tab")
                except LookupError: nltk.download("punkt_tab",quiet=True)
                self._ss=lambda t:[s.strip() for s in nltk.sent_tokenize(t) if s.strip()]
            except ImportError:
                _re=re.compile(r"(?<=[.!?])\s+")
                self._ss=lambda t:[s.strip() for s in _re.split(t) if s.strip()]
        return self._ss(text)
