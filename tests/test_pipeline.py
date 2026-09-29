"""Offline tests: a hashing embedder stands in for the real model."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from app.bm25 import BM25, tokenize
from app.ingest import Page, chunk_page, drop_references, parse_document
from app.llm import build_messages
from app.retriever import HybridRetriever, rrf_fuse
from app.store import IndexStore


class HashEmbedder:
    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, q):
        return self._vec(q)


PAPERS = {
    "attention.md": "Attention Is All You Need\n\n" + " ".join([
        "The Transformer relies entirely on self-attention to compute representations of its input and output.",
        "Multi-head attention lets the model jointly attend to information from different representation subspaces.",
        "On the WMT 2014 English-to-German translation task the big model achieves 28.4 BLEU.",
    ] * 6),
    "resnet.md": "Deep Residual Learning for Image Recognition\n\n" + " ".join([
        "We present a residual learning framework to ease the training of very deep networks.",
        "Shortcut connections perform identity mapping and add neither extra parameters nor computational complexity.",
        "An ensemble of residual nets achieves 3.57% top-5 error on the ImageNet test set.",
    ] * 6),
}


@pytest.fixture
def indexed(tmp_path):
    store = IndexStore(tmp_path / "index", tmp_path / "uploads", "hash")
    emb = HashEmbedder()
    for i, (name, text) in enumerate(PAPERS.items()):
        p = tmp_path / name
        p.write_text(text)
        parsed = parse_document(p, size=60, overlap=15)
        doc = {"id": f"d{i}", "filename": name, "stored_as": name, "title": parsed.title,
               "pages": None, "chunks": len(parsed.chunks), "size_bytes": 0, "sha256": str(i),
               "uploaded_at": f"2026-01-0{i + 1}"}
        store.add(doc, parsed.chunks, emb.encode_docs([c["text"] for c in parsed.chunks]))
    return store, emb


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(100))
    chunks = chunk_page(text, size=50, overlap=12)
    assert len(chunks) > 5
    assert all(len(c.split()) <= 56 for c in chunks)
    # consecutive chunks share their boundary sentence(s)
    assert chunks[0].split(".")[-2] in chunks[1]


def test_references_are_dropped():
    pages = [Page(1, "Intro text"), Page(2, "Method"), Page(3, "Results\nReferences\n[1] Someone 2020")]
    kept = drop_references(pages)
    assert "Someone" not in " ".join(p.text for p in kept)
    assert kept[-1].text.strip() == "Results"


def test_title_is_detected(indexed):
    store, _ = indexed
    titles = {d["title"] for d in store.list_documents()}
    assert "Attention Is All You Need" in titles


def test_bm25_prefers_exact_terms():
    corpus = [tokenize("residual shortcut connections imagenet"), tokenize("self-attention transformer bleu")]
    scores = BM25(corpus).scores(tokenize("transformers BLEU"))
    assert scores[1] > scores[0] == 0


def test_rrf_rewards_agreement():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]])
    assert max(fused, key=fused.get) == 1
    assert fused[3] > fused[2]


def test_hybrid_search_finds_right_paper(indexed):
    store, emb = indexed
    hits = HybridRetriever(store, emb).search("What BLEU score does the Transformer get?", top_k=3)
    assert hits and hits[0]["title"] == "Attention Is All You Need"
    assert [h["ref"] for h in hits] == [1, 2, 3]

    hits = HybridRetriever(store, emb).search("ImageNet top-5 error", top_k=3, doc_ids=["d0"])
    assert all(h["doc_id"] == "d0" for h in hits)


class KeywordReranker:
    """Stand-in cross-encoder: confident only when the passage mentions BLEU."""

    def score(self, query, passages):
        return np.array([4.0 if "BLEU" in p else -4.0 for p in passages], dtype=np.float32)


def test_min_relevance_drops_weak_chunks(indexed):
    store, emb = indexed
    r = HybridRetriever(store, emb, reranker=KeywordReranker())

    hits = r.search("What BLEU score does the Transformer get?", top_k=10, min_relevance=0.3)
    assert hits and all("BLEU" in h["text"] and h["relevance"] >= 0.3 for h in hits)
    assert len(hits) < len(r.search("What BLEU score does the Transformer get?", top_k=10))

    # nothing clears a very high bar -> empty, so the LLM is never asked to improvise
    assert r.search("residual shortcut connections", top_k=6, min_relevance=0.99) == []


def test_store_persists_and_removes(indexed, tmp_path):
    store, _ = indexed
    n = len(store.chunks)
    reloaded = IndexStore(tmp_path / "index", tmp_path / "uploads", "hash")
    assert len(reloaded.chunks) == n and reloaded.embeddings.shape[0] == n

    assert reloaded.remove("d0")
    assert all(c["doc_id"] == "d1" for c in reloaded.chunks)
    assert reloaded.embeddings.shape[0] == len(reloaded.chunks)


def test_model_mismatch_is_caught(indexed, tmp_path):
    with pytest.raises(RuntimeError):
        IndexStore(tmp_path / "index", tmp_path / "uploads", "another-model")


def test_prompt_numbers_sources():
    hits = [{"ref": 1, "title": "Paper A", "page": 3, "text": "alpha"},
            {"ref": 2, "title": "Paper B", "page": None, "text": "beta"}]
    user = build_messages("q?", hits)[1]["content"]
    assert "[1] Paper A, page 3" in user and "[2] Paper B\nbeta" in user


def test_pdf_pages_are_tracked(tmp_path):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    path = tmp_path / "paper.pdf"
    c = canvas.Canvas(str(path))
    for page in range(1, 4):
        c.drawString(72, 750, f"Page {page} discusses gradient descent convergence in detail for our model.")
        c.showPage()
    c.save()
    parsed = parse_document(path, size=60, overlap=10)
    assert [ch["page"] for ch in parsed.chunks] == [1, 2, 3]
