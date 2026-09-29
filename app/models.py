"""Embedding + reranking models. Imported lazily so the rest of the app stays light."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        # bge models expect an instruction prefix on queries only
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([self.query_prefix + query], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec[0], dtype=np.float32)


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance probabilities in [0, 1], one per passage."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        import torch

        pairs = [(query, p) for p in passages]
        # ask for sigmoid explicitly instead of relying on the library default;
        # the kwarg was renamed in sentence-transformers 4
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
