"""Re-run title detection on papers that are already indexed.

    python -m app.retitle          # preview
    python -m app.retitle --apply  # save
"""
import sys

from .config import settings
from .ingest import extract_pages, guess_title
from .store import IndexStore


def main(apply: bool) -> None:
    store = IndexStore(settings.index_dir, settings.upload_dir, settings.embed_model)
    for doc in store.list_documents():
        path = store.file_path(doc["id"])
        if not path or not path.exists():
            continue
        pages, detected = extract_pages(path)
        new = detected or guess_title(pages, doc["filename"].rsplit(".", 1)[0])
        if new != doc["title"]:
            print(f"{doc['filename']}\n  - {doc['title']}\n  + {new}")
            if apply:
                store.update(doc["id"], title=new)
    if not apply:
        print("\n(preview only — add --apply to save)")


if __name__ == "__main__":
    main("--apply" in sys.argv)
