"""Embedding + reranking models. Imported lazily so the rest of the app stays light."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np

from .config import settings

# HF tokenizers' own thread pool can deadlock when called from background threads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# uploads, searches and re-indexing run on different threads; torch (MPS especially)
# isn't safe to call concurrently, so every model call goes through this lock
_MODEL_LOCK = threading.Lock()

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=settings.device)   # None = auto (mps/cuda/cpu)
        self.device = str(self.model.device)
        # bge models expect an instruction prefix on queries only
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        with _MODEL_LOCK:
            vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        with _MODEL_LOCK:
            vec = self.model.encode([self.query_prefix + query], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec[0], dtype=np.float32)


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name, max_length=512, device=settings.device)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance probabilities in [0, 1], one per passage."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        import torch

        pairs = [(query, p) for p in passages]
        # ask for sigmoid explicitly instead of relying on the library default;
        # the kwarg was renamed in sentence-transformers 4
        with _MODEL_LOCK:
            try:
                probs = self.model.predict(pairs, batch_size=16, activation_fn=torch.nn.Sigmoid())
            except TypeError:
                probs = self.model.predict(pairs, batch_size=16, activation_fct=torch.nn.Sigmoid())
        return np.asarray(probs, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(settings.embed_model)


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    return Reranker(settings.rerank_model) if settings.use_reranker else None
