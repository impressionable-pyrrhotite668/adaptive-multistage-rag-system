"""
embedder.py — Batched Sentence-Transformer Embeddings
Provides an Embedder CLASS (wraps standalone functions for compatibility).
"""
import numpy as np

_MODEL_CACHE = {}

def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        print(f"[Embedder] Loading model: {model_name}")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class Embedder:
    """Class-based wrapper so main.py can use Embedder() instances."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 batch_size=64, normalize=True, device=None, cache_dir=None):
        # Strip "sentence-transformers/" prefix if present — HuggingFace handles it
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize  = normalize
        self._model     = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedder] Loading: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self._get().get_sentence_embedding_dimension()

    def embed_texts(self, texts: list) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype="float32")
        return self._get().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: list):
        """Returns (chunks, embeddings_matrix)."""
        texts = [c.text for c in chunks]
        embs  = self.embed_texts(texts)
        return chunks, embs


# Keep standalone functions for backward compatibility
def embed_texts(texts, model_name="all-MiniLM-L6-v2", batch_size=64, normalize=True):
    return Embedder(model_name=model_name, batch_size=batch_size,
                    normalize=normalize).embed_texts(texts)

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2", batch_size=64, normalize=True):
    e = Embedder(model_name=model_name, batch_size=batch_size, normalize=normalize)
    return e.embed_chunks(chunks)

def get_embedding_dim(model_name="all-MiniLM-L6-v2"):
    return Embedder(model_name=model_name).embedding_dim
