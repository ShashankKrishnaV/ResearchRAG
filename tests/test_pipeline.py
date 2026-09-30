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
    """Stand-in cross-encoder: returns probabilities, like the real wrapper."""

    def score(self, query, passages):
        return np.array([0.9 if "BLEU" in p else 0.05 for p in passages], dtype=np.float32)


def test_min_relevance_drops_weak_chunks(indexed):
    store, emb = indexed
    r = HybridRetriever(store, emb, reranker=KeywordReranker())

    hits = r.search("What BLEU score does the Transformer get?", top_k=10, min_relevance=0.3)
    assert hits and all("BLEU" in h["text"] for h in hits)
    # probabilities pass through untouched (no second sigmoid squashing them toward 0.5-0.73)
    assert {h["relevance"] for h in hits} == {0.9}
    weak = r.search("What BLEU score does the Transformer get?", top_k=10)
    assert min(h["relevance"] for h in weak) == 0.05
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


def test_mismatch_can_be_tolerated_for_the_web_app(indexed, tmp_path):
    store = IndexStore(tmp_path / "index", tmp_path / "uploads", "another-model", allow_mismatch=True)
    assert store.stale_model == "hash" and store.stats()["documents"] == 2


def test_rebuild_reembeds_and_keeps_titles(indexed, tmp_path):
    from app.reindex import rebuild

    store, emb = indexed
    store.update("d0", title="My renamed title")
    for doc in store.list_documents():            # rebuild re-reads the original files from uploads/
        (tmp_path / doc["filename"]).rename(tmp_path / "uploads" / doc["stored_as"])

    seen = []
    new = rebuild(tmp_path / "index", tmp_path / "uploads", "new-model", emb, 60, 15,
                  progress=lambda done, total, title: seen.append((done, total)))

    assert new.stale_model is None and new.embed_model == "new-model"
    assert new.documents["d0"]["title"] == "My renamed title"
    assert len(new.chunks) == len(store.chunks) and new.embeddings.shape[0] == len(new.chunks)
    assert seen[-1] == (2, 2)
    assert not list(tmp_path.glob("index.bak-*"))  # backup cleaned up
    IndexStore(tmp_path / "index", tmp_path / "uploads", "new-model")   # loads without complaint


def test_failed_rebuild_restores_old_index(indexed, tmp_path):
    from app.reindex import rebuild

    store, _ = indexed
    for doc in store.list_documents():
        (tmp_path / doc["filename"]).rename(tmp_path / "uploads" / doc["stored_as"])

    class Broken:
        def encode_docs(self, texts):
            raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError):
        rebuild(tmp_path / "index", tmp_path / "uploads", "new-model", Broken(), 60, 15)
    restored = IndexStore(tmp_path / "index", tmp_path / "uploads", "hash")
    assert restored.stats() == store.stats()


def test_interrupted_rebuild_is_recovered(indexed, tmp_path):
    from app.reindex import recover_interrupted

    store, _ = indexed
    index = tmp_path / "index"
    index.rename(tmp_path / "index.bak-123")          # what rebuild() does first...
    IndexStore(index, tmp_path / "uploads", "new-model")   # ...then the server dies mid-build

    assert recover_interrupted(index)
    assert IndexStore(index, tmp_path / "uploads", "hash").stats() == store.stats()
    assert not list(tmp_path.glob("index.bak-*"))
    assert not recover_interrupted(index)             # nothing to do the second time


def test_prompt_numbers_sources():
    hits = [{"ref": 1, "title": "Paper A", "page": 3, "text": "alpha"},
            {"ref": 2, "title": "Paper B", "page": None, "text": "beta"}]
    user = build_messages("q?", hits)[1]["content"]
    assert "[1] Paper A, page 3" in user and "[2] Paper B\nbeta" in user


def _title_pdf(path, lines, header=None, meta=None):
    """lines: list of title lines, each a list of (size, text) runs."""
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    c = canvas.Canvas(str(path))
    if meta:
        c.setTitle(meta)
    if header:
        c.setFont("Times-Roman", 9)
        c.drawString(72, 780, header)
    c.saveState()   # rotated arXiv-style stamp in a big font, drawn via the page transform
    c.setFont("Times-Roman", 20); c.translate(30, 300); c.rotate(90)
    c.drawString(0, 0, "arXiv:2010.11929v2 [cs.CV] 3 Jun 2021")
    c.restoreState()
    y = 700
    for line in lines:
        t = c.beginText(110, y)
        for size, text in line:
            t.setFont("Times-Bold", size)
            t.textOut(text)
        c.drawText(t)
        y -= max(s for s, _ in line) * 1.3
    c.setFont("Times-Roman", 11)
    c.drawString(72, y - 20, "First Author, Second Author  Some University")
    c.save()
    return parse_document(path, size=220, overlap=40).title


@pytest.mark.parametrize("lines, header, meta, expected", [
    ([[(17, "Attention Is All You Need")]],
     "Provided proper attribution is provided, Google hereby grants permission to", None,
     "Attention Is All You Need"),
    ([[(17, "A"), (13, "N "), (17, "I"), (13, "MAGE IS "), (17, "W"), (13, "ORTH "), (17, "16"), (13, "X"), (17, "16 W"), (13, "ORDS")]],
     "Published as a conference paper at ICLR 2021", None,
     "AN IMAGE IS WORTH 16X16 WORDS"),                              # small caps
    ([[(18, "Deep Residual Learning")], [(15.5, "for Image Recognition")]], None, None,
     "Deep Residual Learning for Image Recognition"),               # wrapped, second line smaller
    ([[(17, "Language Models are Few-Shot Learners")]], None, "arXiv preprint",
     "Language Models are Few-Shot Learners"),                      # junk metadata ignored
    ([[(17, "Something Else")]], None, "Denoising Diffusion Probabilistic Models",
     "Denoising Diffusion Probabilistic Models"),                   # good metadata still wins
])
def test_pdf_title_detection(tmp_path, lines, header, meta, expected):
    assert _title_pdf(tmp_path / "t.pdf", lines, header, meta) == expected


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
