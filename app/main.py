import hashlib
import json
import logging
import mimetypes
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT, settings
from .ingest import SUPPORTED, parse_document
from .llm import LLMUnavailable, llm_status, stream_answer
from .models import get_embedder, get_reranker
from .retriever import HybridRetriever
from .store import IndexStore

log = logging.getLogger("researchrag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

WEB = ROOT / "web"
MAX_UPLOAD_MB = 100

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("loading embedder: %s", settings.embed_model)
    embedder = get_embedder()
    log.info("loading reranker: %s", settings.rerank_model if settings.use_reranker else "disabled")
    reranker = get_reranker()

    store = IndexStore(settings.index_dir, settings.upload_dir, settings.embed_model)
    state["store"] = store
    state["embedder"] = embedder
    state["retriever"] = HybridRetriever(store, embedder, reranker, settings.candidates)
    log.info("index ready: %s", store.stats())
    yield


app = FastAPI(title="ResearchRAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


class QueryIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=settings.top_k, ge=1, le=15)
    doc_ids: list[str] | None = None


# ---------- pages ----------

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/ask")


@app.get("/library", include_in_schema=False)
def library_page():
    return FileResponse(WEB / "library.html")


@app.get("/ask", include_in_schema=False)
def ask_page():
    return FileResponse(WEB / "ask.html")


# ---------- documents ----------

def _ingest(path: Path, filename: str, sha: str, size: int) -> dict:
    parsed = parse_document(path, settings.chunk_words, settings.chunk_overlap, display_name=filename)
    if not parsed.chunks:
        raise ValueError("No extractable text found (is it a scanned PDF?)")

    vectors = state["embedder"].encode_docs([c["text"] for c in parsed.chunks])
    doc = {
        "id": path.stem,
        "filename": filename,
        "stored_as": path.name,
        "title": parsed.title,
        "pages": len(parsed.pages) if parsed.pages and parsed.pages[0].number else None,
        "chunks": len(parsed.chunks),
        "size_bytes": size,
        "sha256": sha,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    state["store"].add(doc, parsed.chunks, vectors)
    return doc


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    store: IndexStore = state["store"]
    results = []

    for f in files:
        name = Path(f.filename or "untitled").name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED:
            results.append({"filename": name, "status": "error", "error": f"Unsupported type {ext or '(none)'}"})
            continue

        data = await f.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            results.append({"filename": name, "status": "error", "error": f"File is over {MAX_UPLOAD_MB} MB"})
            continue

        sha = hashlib.sha256(data).hexdigest()
        if existing := store.find_by_hash(sha):
            results.append({"filename": name, "status": "duplicate", "doc": existing})
            continue

        path = settings.upload_dir / f"{uuid.uuid4().hex[:12]}{ext}"
        path.write_bytes(data)
        t0 = time.perf_counter()
        try:
            doc = await run_in_threadpool(_ingest, path, name, sha, len(data))
            log.info("indexed %s → %d chunks in %.1fs", name, doc["chunks"], time.perf_counter() - t0)
            results.append({"filename": name, "status": "indexed", "doc": doc})
        except Exception as e:
            path.unlink(missing_ok=True)
            log.warning("failed to index %s: %s", name, e)
            results.append({"filename": name, "status": "error", "error": str(e)})

    return {"results": results, "stats": store.stats()}


@app.get("/api/documents")
def list_documents():
    store: IndexStore = state["store"]
    return {"documents": store.list_documents(), "stats": store.stats()}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if not state["store"].remove(doc_id):
        raise HTTPException(404, "Document not found")
    return {"ok": True, "stats": state["store"].stats()}


@app.get("/api/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    store: IndexStore = state["store"]
    path = store.file_path(doc_id)
    if not path or not path.exists():
        raise HTTPException(404, "File not found")
    doc = store.documents[doc_id]
    media = mimetypes.guess_type(doc["filename"])[0] or "application/octet-stream"
    return FileResponse(path, media_type=media, filename=doc["filename"], content_disposition_type="inline")


# ---------- retrieval + generation ----------

@app.post("/api/search")
async def search(q: QueryIn):
    t0 = time.perf_counter()
    hits = await run_in_threadpool(state["retriever"].search, q.question, q.top_k, q.doc_ids)
    return {"hits": hits, "elapsed_ms": int((time.perf_counter() - t0) * 1000)}


@app.post("/api/ask")
async def ask(q: QueryIn):
    t0 = time.perf_counter()
    hits = await run_in_threadpool(state["retriever"].search, q.question, q.top_k, q.doc_ids)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    def events():
        def emit(obj):
            return json.dumps(obj, ensure_ascii=False) + "\n"

        yield emit({"type": "sources", "hits": hits, "retrieval_ms": retrieval_ms})
        if not hits:
            yield emit({"type": "error", "message": "Nothing to search yet — upload some papers in the Library first."})
            return
        try:
            for token in stream_answer(q.question, hits):
                yield emit({"type": "token", "text": token})
        except LLMUnavailable as e:
            yield emit({"type": "error", "message": str(e)})
        yield emit({"type": "done", "total_ms": int((time.perf_counter() - t0) * 1000)})

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/health")
def health():
    return {
        "index": state["store"].stats(),
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model if settings.use_reranker else None,
        "llm": llm_status(),
    }
