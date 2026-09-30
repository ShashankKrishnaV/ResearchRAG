"""Rebuild the index with the embedding model currently set in config/.env.

Needed after changing RAG_EMBED_MODEL (vectors from different models can't be mixed).
Either click "Rebuild index" in the Library page, or stop the server and run:

    python -m app.reindex
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from .ingest import parse_document
from .store import IndexStore

# progress(done, total, current_title)
Progress = Callable[[int, int, str], None]


def recover_interrupted(index_dir: Path) -> bool:
    """If a rebuild was killed midway, a backup is still lying around — put it back."""
    index_dir = Path(index_dir)
    backups = sorted(index_dir.parent.glob(f"{index_dir.name}.bak-*"))
    if not backups:
        return False
    shutil.rmtree(index_dir, ignore_errors=True)       # half-built index
    shutil.move(str(backups[-1]), str(index_dir))
    for extra in backups[:-1]:
        shutil.rmtree(extra, ignore_errors=True)
    return True


def rebuild(index_dir: Path, upload_dir: Path, embed_model: str, embedder,
            chunk_words: int, chunk_overlap: int, progress: Optional[Progress] = None) -> IndexStore:
    """Re-parse and re-embed every stored paper. Shared by the CLI and the web UI."""
    index_dir, upload_dir = Path(index_dir), Path(upload_dir)
    docs_file = index_dir / "documents.json"
    docs = json.loads(docs_file.read_text()).get("documents", []) if docs_file.exists() else []

    # keep the old index until the new one is fully built
    backup = index_dir.with_name(f"index.bak-{int(time.time())}")
    if index_dir.exists():
        shutil.move(str(index_dir), str(backup))

    try:
        store = IndexStore(index_dir, upload_dir, embed_model)
        for n, doc in enumerate(docs):
            if progress:
                progress(n, len(docs), doc["title"])
            path = upload_dir / doc["stored_as"]
            if not path.exists():
                continue
            parsed = parse_document(path, chunk_words, chunk_overlap, display_name=doc["filename"])
            vectors = embedder.encode_docs([c["text"] for c in parsed.chunks])
            # keep id and title (the user may have renamed it), refresh the chunk count
            store.add({**doc, "chunks": len(parsed.chunks)}, parsed.chunks, vectors)
        if progress:
            progress(len(docs), len(docs), "")
    except Exception:
        shutil.rmtree(index_dir, ignore_errors=True)
        if backup.exists():
            shutil.move(str(backup), str(index_dir))
        raise

    shutil.rmtree(backup, ignore_errors=True)
    return store


def main() -> None:
    from .config import settings
    from .models import get_embedder

    def show(done, total, title):
        print(f"  [{done}/{total}] {title[:60]}" if title else f"  [{done}/{total}] done")

    if recover_interrupted(settings.index_dir):
        print("Restored the index from an interrupted rebuild.")
    print(f"Re-indexing with {settings.embed_model}")
    try:
        store = rebuild(settings.index_dir, settings.upload_dir, settings.embed_model, get_embedder(),
                        settings.chunk_words, settings.chunk_overlap, progress=show)
    except Exception:
        print("Failed — old index restored.")
        raise
    print(f"All set: {store.stats()}. Start the app with ./run.sh")


if __name__ == "__main__":
    main()
