"""Hybrid retrieval: dense + BM25 -> reciprocal rank fusion -> cross-encoder rerank."""
import math

import numpy as np

from .bm25 import tokenize
from .store import IndexStore

RRF_K = 60


def _top_indices(scores: np.ndarray, allowed: np.ndarray, n: int) -> list[int]:
    s = np.where(allowed, scores, -np.inf)
    n = min(n, int(allowed.sum()))
    if n <= 0:
        return []
    idx = np.argpartition(-s, n - 1)[:n]
    idx = idx[np.argsort(-s[idx])]
    return [int(i) for i in idx if np.isfinite(s[i])]


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, store: IndexStore, embedder, reranker=None, candidates: int = 30):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, top_k: int = 6, doc_ids: list[str] | None = None) -> list[dict]:
        store = self.store
        if not store.chunks or not query.strip():
            return []

        if doc_ids:
            wanted = set(doc_ids)
            allowed = np.array([c["doc_id"] in wanted for c in store.chunks])
        else:
            allowed = np.ones(len(store.chunks), dtype=bool)

        # 1) dense: vectors are normalized, so dot product == cosine
        dense_scores = store.embeddings @ self.embedder.encode_query(query)
        dense_rank = _top_indices(dense_scores, allowed, self.candidates)

        # 2) sparse: ignore chunks with zero keyword overlap
        bm25_scores = store.bm25.scores(tokenize(query))
        bm25_rank = _top_indices(bm25_scores, allowed & (bm25_scores > 0), self.candidates)

        # 3) fuse the two rankings
        fused = rrf_fuse([dense_rank, bm25_rank])
        pool = sorted(fused, key=fused.get, reverse=True)[: self.candidates]

        # 4) rerank the pool with the cross-encoder
        if self.reranker is not None and pool:
            logits = self.reranker.score(query, [store.chunks[i]["text"] for i in pool])
            order = np.argsort(-logits)
            ranked = [(pool[j], 1 / (1 + math.exp(-float(logits[j])))) for j in order]
        else:
            top = fused[pool[0]] if pool else 1.0
            ranked = [(i, fused[i] / top) for i in pool]

        d_pos = {idx: r + 1 for r, idx in enumerate(dense_rank)}
        b_pos = {idx: r + 1 for r, idx in enumerate(bm25_rank)}

        hits = []
        for idx, relevance in ranked[:top_k]:
            c = store.chunks[idx]
            doc = store.documents.get(c["doc_id"], {})
            hits.append({
                "ref": len(hits) + 1,
                "chunk_id": c["id"],
                "doc_id": c["doc_id"],
                "title": doc.get("title", "Untitled"),
                "filename": doc.get("filename", ""),
                "is_pdf": doc.get("filename", "").lower().endswith(".pdf"),
                "page": c["page"],
                "text": c["text"],
                "relevance": round(relevance, 4),
                "dense_rank": d_pos.get(idx),
                "bm25_rank": b_pos.get(idx),
            })
        return hits
