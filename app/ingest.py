"""Turn uploaded files into clean, page-aware chunks."""
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}

_REFS_HEADING = re.compile(r"^\s*(references|bibliography|works cited)\s*$", re.I | re.M)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")


@dataclass
class Page:
    number: int | None  # None for formats without real pages (docx, txt)
    text: str


@dataclass
class ParsedDoc:
    title: str
    pages: list[Page]
    chunks: list[dict] = field(default_factory=list)


# ---------- extraction ----------

def _read_pdf(path: Path) -> tuple[list[Page], str | None]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [Page(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]

    meta_title = None
    try:
        t = (reader.metadata or {}).get("/Title")
        if t and len(str(t).strip()) > 5 and not str(t).lower().startswith(("microsoft word", "untitled")):
            meta_title = str(t).strip()
    except Exception:
        pass
    return pages, meta_title


def _read_docx(path: Path) -> list[Page]:
    import docx

    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    return [Page(None, text)]


def extract_pages(path: Path) -> tuple[list[Page], str | None]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path), None
    if ext in {".txt", ".md"}:
        return [Page(None, path.read_text(errors="ignore"))], None
    raise ValueError(f"Unsupported file type: {ext}")


# ---------- cleanup ----------

def guess_title(pages: list[Page], fallback: str) -> str:
    # first reasonably sized line on the first page is usually the title
    if pages:
        for line in pages[0].text.splitlines()[:12]:
            line = line.strip().lstrip("#").strip()
            words = line.split()
            if 3 <= len(words) <= 25 and not re.search(r"arxiv|@|doi|preprint|vol\.", line, re.I):
                return line
    return fallback


def drop_references(pages: list[Page]) -> list[Page]:
    # cut everything after a "References" heading found in the back half of the paper
    if len(pages) < 2:
        return pages
    for i in range(len(pages) - 1, len(pages) // 2 - 1, -1):
        m = None
        for m in _REFS_HEADING.finditer(pages[i].text):
            pass
        if m:
            kept = pages[:i]
            head = pages[i].text[: m.start()]
            if head.strip():
                kept.append(Page(pages[i].number, head))
            return kept
    return pages


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join hyphenated words
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)          # pdf line breaks -> spaces
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


# ---------- chunking ----------

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _word_windows(sentence: str, size: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def chunk_page(text: str, size: int, overlap: int) -> list[str]:
    sentences = []
    for s in split_sentences(text):
        # very long "sentences" (tables, equations) get hard-split
        sentences.extend(_word_windows(s, size) if len(s.split()) > size else [s])

    chunks, buf, buf_len = [], [], 0
    for s in sentences:
        n = len(s.split())
        if buf and buf_len + n > size:
            chunks.append(" ".join(buf))
            # carry a few trailing sentences forward as overlap
            tail, tail_len = [], 0
            for prev in reversed(buf):
                m = len(prev.split())
                if tail_len + m > overlap * 2:
                    break
                tail.insert(0, prev)
                tail_len += m
                if tail_len >= overlap:
                    break
            if tail_len + n > size:
                tail, tail_len = [], 0
            buf, buf_len = tail, tail_len
        buf.append(s)
        buf_len += n
    if buf:
        chunks.append(" ".join(buf))

    # drop tiny fragments (page headers, stray numbers)
    return [c for c in chunks if len(c.split()) >= 8]


def parse_document(path: Path, size: int, overlap: int, display_name: str | None = None) -> ParsedDoc:
    raw_pages, meta_title = extract_pages(path)
    name = display_name or path.name
    title = meta_title or guess_title(raw_pages, Path(name).stem)

    pages = drop_references(raw_pages)
    doc = ParsedDoc(title=title, pages=raw_pages)
    for page in pages:
        for text in chunk_page(clean_text(page.text), size, overlap):
            doc.chunks.append({"text": text, "page": page.number})
    return doc
