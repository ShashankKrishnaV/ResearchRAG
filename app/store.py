"""On-disk index: chunk texts (jsonl), vectors (npy) and a small document registry (json)."""
import json
import os
import threading
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize


def _atomic_write(path: Path, write_fn) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


class IndexStore:
    def __init__(self, index_dir: Path, upload_dir: Path, embed_model: str):
        self.index_dir, self.upload_dir = Path(index_dir), Path(upload_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model

        self._docs_path = self.index_dir / "documents.json"
        self._chunks_path = self.index_dir / "chunks.jsonl"
        self._emb_path = self.index_dir / "embeddings.npy"

        self._lock = threading.RLock()
        self.documents: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._bm25: BM25 | None = None
        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if self._docs_path.exists():
            data = json.loads(self._docs_path.read_text())
            saved_model = data.get("embed_model")
            if saved_model and saved_model != self.embed_model:
                raise RuntimeError(
                    f"Index was built with '{saved_model}' but config uses '{self.embed_model}'. "
                    "Delete data/index/ and re-upload, or switch the model back."
                )
            self.documents = {d["id"]: d for d in data.get("documents", [])}
        if self._chunks_path.exists():
            with self._chunks_path.open() as f:
                self.chunks = [json.loads(line) for line in f if line.strip()]
        if self._emb_path.exists():
            self.embeddings = np.load(self._emb_path)

        if self.embeddings is not None and len(self.embeddings) != len(self.chunks):
            raise RuntimeError("Index is out of sync (chunks vs embeddings). Delete data/index/ and re-upload.")

    def _save(self) -> None:
        payload = {"embed_model": self.embed_model, "documents": list(self.documents.values())}
        _atomic_write(self._docs_path, lambda p: p.write_text(json.dumps(payload, indent=2)))

        def write_chunks(p: Path):
            with p.open("w") as f:
                for c in self.chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

        _atomic_write(self._chunks_path, write_chunks)

        if self.embeddings is not None:
            def write_emb(p: Path):
                with p.open("wb") as f:
                    np.save(f, self.embeddings)
            _atomic_write(self._emb_path, write_emb)
        elif self._emb_path.exists():
            self._emb_path.unlink()

    # ---------- reads ----------

    @property
    def bm25(self) -> BM25:
        with self._lock:
            if self._bm25 is None:
                self._bm25 = BM25([tokenize(c["text"]) for c in self.chunks])
            return self._bm25

    def list_documents(self) -> list[dict]:
        return sorted(self.documents.values(), key=lambda d: d["uploaded_at"], reverse=True)

    def find_by_hash(self, sha256: str) -> dict | None:
        return next((d for d in self.documents.values() if d["sha256"] == sha256), None)

    def file_path(self, doc_id: str) -> Path | None:
        doc = self.documents.get(doc_id)
        return self.upload_dir / doc["stored_as"] if doc else None

    def stats(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}

    # ---------- writes ----------

    def add(self, doc: dict, chunks: list[dict], vectors: np.ndarray) -> None:
        assert len(chunks) == len(vectors), "one vector per chunk"
        with self._lock:
            for i, c in enumerate(chunks):
                self.chunks.append({"id": f"{doc['id']}:{i}", "doc_id": doc["id"], "page": c["page"], "text": c["text"]})
            vectors = vectors.astype(np.float32)
            self.embeddings = vectors if self.embeddings is None else np.vstack([self.embeddings, vectors])
            self.documents[doc["id"]] = doc
            self._bm25 = None
            self._save()

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            doc = self.documents.pop(doc_id, None)
            if not doc:
                return False
            keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            if self.embeddings is not None:
                self.embeddings = self.embeddings[keep] if keep else None
            self._bm25 = None
            self._save()

            f = self.upload_dir / doc["stored_as"]
            if f.exists():
                f.unlink()
            return True
