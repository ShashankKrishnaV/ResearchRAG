"""Rebuild the index with the embedding model currently set in config/.env.

Needed after changing RAG_EMBED_MODEL (vectors from different models can't be mixed).
Stop the server first, then:

    python -m app.reindex
"""
import json
import shutil
import time

from .config import settings
from .ingest import parse_document
from .models import get_embedder
from .store import IndexStore


def main() -> None:
    index_dir = settings.index_dir
    docs_file = index_dir / "documents.json"
    if not docs_file.exists():
        print("Nothing indexed yet — just start the app.")
        return

    old = json.loads(docs_file.read_text())
    docs = old.get("documents", [])
    print(f"Re-indexing {len(docs)} papers: {old.get('embed_model')} -> {settings.embed_model}")

    # keep the old index until the new one is fully built
    backup = index_dir.with_name(f"index.bak-{int(time.time())}")
    shutil.move(str(index_dir), str(backup))

    try:
        embedder = get_embedder()
        store = IndexStore(index_dir, settings.upload_dir, settings.embed_model)
        for doc in docs:
            path = settings.upload_dir / doc["stored_as"]
            if not path.exists():
                print(f"  skip  {doc['filename']} (file missing)")
                continue
            parsed = parse_document(path, settings.chunk_words, settings.chunk_overlap, display_name=doc["filename"])
            vectors = embedder.encode_docs([c["text"] for c in parsed.chunks])
            # keep id and title (the user may have renamed it), refresh the chunk count
            store.add({**doc, "chunks": len(parsed.chunks)}, parsed.chunks, vectors)
            print(f"  done  {doc['title'][:60]}  ({len(parsed.chunks)} chunks)")
    except Exception:
        shutil.rmtree(index_dir, ignore_errors=True)
        shutil.move(str(backup), str(index_dir))
        print("Failed — old index restored.")
        raise

    shutil.rmtree(backup)
    print("All set. Start the app with ./run.sh")


if __name__ == "__main__":
    main()
