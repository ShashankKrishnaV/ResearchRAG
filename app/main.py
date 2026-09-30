import hashlib
import json
import logging
import mimetypes
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT, settings
from .ingest import SUPPORTED, parse_document
from .llm import LLMUnavailable, list_models, llm_status, stream_answer
from .models import get_embedder, get_reranker
from .reindex import rebuild, recover_interrupted
from .retriever import HybridRetriever
from .store import IndexStore

log = logging.getLogger("researchrag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

WEB = ROOT / "web"
MAX_UPLOAD_MB = 100

state: dict = {}
_job_lock = threading.Lock()


def _install(store: IndexStore) -> None:
    state["store"] = store
    state["retriever"] = HybridRetriever(store, state["embedder"], state["reranker"], settings.candidates)


def _open_store() -> IndexStore:
    # allow_mismatch: if the embedding model changed, start anyway and offer a rebuild in the UI
    return IndexStore(settings.index_dir, settings.upload_dir, settings.embed_model, allow_mismatch=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("loading embedder: %s", settings.embed_model)
    state["embedder"] = get_embedder()
    log.info("models running on: %s", state["embedder"].device)
    log.info("loading reranker: %s", settings.rerank_model if settings.use_reranker else "disabled")
    state["reranker"] = get_reranker()
    state["job"] = {"running": False, "done": 0, "total": 0, "current": "", "error": None}

    if recover_interrupted(settings.index_dir):
        log.warning("previous rebuild was interrupted — restored the old index")
        state["job"]["error"] = "The previous rebuild was interrupted (server stopped?). Your old index was restored."

    _install(_open_store())
    store = state["store"]
    if store.stale_model:
        log.warning("index was built with %s, config says %s — rebuild it from the Library page",
                    store.stale_model, settings.embed_model)
    log.info("index ready: %s", store.stats())
    yield


app = FastAPI(title="ResearchRAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


class QueryIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=settings.top_k, ge=1, le=15)
    doc_ids: Optional[List[str]] = None  # pydantic evaluates this at runtime, so no `|` on 3.9
    min_relevance: float = Field(default=settings.min_relevance, ge=0, le=1)
    llm_model: Optional[str] = Field(default=None, max_length=100)   # None -> configured default


# ---------- guards ----------

def _require_idle() -> None:
    if state["job"]["running"]:
        raise HTTPException(409, "The index is being rebuilt — try again in a moment.")


def _require_ready() -> None:
    # searching or adding papers needs vectors from the current embedding model
    _require_idle()
    stale = state["store"].stale_model
    if stale:
        raise HTTPException(409, f"Your library was indexed with {stale}, but the app is now set to "
                                 f"{settings.embed_model}. Rebuild the index from the Library page first.")


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
async def upload_documents(files: List[UploadFile] = File(...)):
    _require_ready()
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


class DocPatch(BaseModel):
    title: str = Field(min_length=1, max_length=300)


@app.patch("/api/documents/{doc_id}")
def rename_document(doc_id: str, patch: DocPatch):
    _require_idle()
    doc = state["store"].update(doc_id, title=patch.title.strip())
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    _require_idle()
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


# ---------- re-index (after changing the embedding model or chunk settings) ----------

def _reindex_status() -> dict:
    store: IndexStore = state["store"]
    return {
        **state["job"],
        "now": time.time(),   # lets the page tell "slow" from "stuck" without trusting the browser clock
        "needs_reindex": bool(store.stale_model),
        "index_model": store.stale_model or store.embed_model,
        "config_model": settings.embed_model,
        "documents": len(store.documents),
    }


def _run_reindex() -> None:
    job = state["job"]
    last = [time.perf_counter()]

    def progress(done, total, title):
        now = time.perf_counter()
        if done:
            log.info("re-index %d/%d done (%.1fs)", done, total, now - last[0])
        if title:
            log.info("re-index %d/%d: %s", done + 1, total, title[:70])
        last[0] = now
        job.update(done=done, total=total, current=title, updated_at=time.time())

    try:
        t0 = time.perf_counter()
        new_store = rebuild(settings.index_dir, settings.upload_dir, settings.embed_model, state["embedder"],
                            settings.chunk_words, settings.chunk_overlap, progress=progress)
        _install(new_store)
        log.info("re-index finished: %s in %.1fs", new_store.stats(), time.perf_counter() - t0)
    except Exception as e:
        log.exception("re-index failed")
        job["error"] = str(e)
        _install(_open_store())   # rebuild() restored the old files; reload them
    finally:
        job.update(running=False, current="")


@app.get("/api/reindex")
def reindex_status():
    return _reindex_status()


@app.post("/api/reindex")
def start_reindex():
    with _job_lock:
        if state["job"]["running"]:
            raise HTTPException(409, "A rebuild is already running.")
        total = len(state["store"].documents)
        now = time.time()
        state["job"].update(running=True, done=0, total=total, current="", error=None,
                            started_at=now, updated_at=now)
    threading.Thread(target=_run_reindex, name="reindex", daemon=True).start()
    return _reindex_status()


# ---------- retrieval + generation ----------

def _retrieve(q: QueryIn) -> list[dict]:
    return state["retriever"].search(q.question, q.top_k, q.doc_ids, q.min_relevance)


def _empty_reason(q: QueryIn) -> str:
    if not state["store"].chunks:
        return "Nothing to search yet — upload some papers in the Library first."
    return (f"No passage reached {q.min_relevance:.0%} relevance for this question. "
            "Try rephrasing, widening the paper filter, or lowering the relevance threshold.")


@app.post("/api/search")
async def search(q: QueryIn):
    _require_ready()
    t0 = time.perf_counter()
    hits = await run_in_threadpool(_retrieve, q)
    return {
        "hits": hits,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "message": None if hits else _empty_reason(q),
    }


@app.post("/api/ask")
async def ask(q: QueryIn):
    _require_ready()
    t0 = time.perf_counter()
    hits = await run_in_threadpool(_retrieve, q)
    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    model = q.llm_model or settings.llm_model

    def events():
        def emit(obj):
            return json.dumps(obj, ensure_ascii=False) + "\n"

        yield emit({"type": "sources", "hits": hits, "retrieval_ms": retrieval_ms, "model": model})
        if not hits:
            yield emit({"type": "error", "message": _empty_reason(q)})   # don't let the LLM answer from nothing
            return
        try:
            for token in stream_answer(q.question, hits, model=model):
                yield emit({"type": "token", "text": token})
        except LLMUnavailable as e:
            yield emit({"type": "error", "message": str(e)})
        yield emit({"type": "done", "total_ms": int((time.perf_counter() - t0) * 1000)})

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/llm/models")
def llm_models():
    return {"models": list_models(), "default": settings.llm_model}


@app.get("/api/health")
def health():
    return {
        "index": state["store"].stats(),
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model if settings.use_reranker else None,
        "min_relevance": settings.min_relevance,
        "needs_reindex": bool(state["store"].stale_model),
        "reindexing": state["job"]["running"],
        "llm": llm_status(),
    }
