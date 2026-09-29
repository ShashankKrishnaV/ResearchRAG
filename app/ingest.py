"""Turn uploaded files into clean, page-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}

_REFS_HEADING = re.compile(r"^\s*(references|bibliography|works cited)\s*$", re.I | re.M)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
# running headers, licence notices and venue lines that sit above the real title
_BOILERPLATE = re.compile(
    r"arxiv|preprint|published as|conference paper|under review|proceedings|workshop|journal of|"
    r"copyright|©|licen[cs]e|permission|attribution|all rights|\bdoi\b|https?://|@|vol\.|issn",
    re.I,
)


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

def _title_by_font(page) -> str | None:
    # the title is almost always the biggest horizontal text on page 1
    runs = []

    def visit(text, cm, tm, font_dict, font_size):
        t = " ".join(text.split())
        if not t:
            return
        if abs(tm[1]) > 0.01 or abs(tm[2]) > 0.01:   # rotated text, e.g. the arXiv side stamp
            return
        scale = abs(tm[3] * cm[3]) or 1.0
        runs.append((round(font_size * scale, 1), t))

    try:
        page.extract_text(visitor_text=visit)
    except Exception:
        return None

    usable = [(s, t) for s, t in runs if sum(c.isalpha() for c in t) >= 2 and not _BOILERPLATE.search(t)]
    if not usable:
        return None
    top = max(s for s, _ in usable)

    # take the first contiguous block of largest-font runs (multi-line titles)
    parts = []
    for s, t in runs:
        if abs(s - top) <= 0.6:
            parts.append(t)
        elif parts and len(" ".join(parts)) >= 10:
            break
    title = " ".join(" ".join(parts).split()).strip(" *†‡∗")
    return title if _looks_like_title(title) else None


def _read_pdf(path: Path) -> tuple[list[Page], str | None]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [Page(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]

    title = None
    try:
        t = str((reader.metadata or {}).get("/Title") or "").strip()
        if _looks_like_title(t) and not re.search(r"\.(dvi|pdf|tex|docx?)$|^microsoft word|^untitled", t, re.I):
            title = t
    except Exception:
        pass
    if not title and reader.pages:
        title = _title_by_font(reader.pages[0])
    return pages, title


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

def _looks_like_title(text: str) -> bool:
    words = text.split()
    return 2 <= len(words) <= 30 and sum(c.isalpha() for c in text) >= 8


def guess_title(pages: list[Page], fallback: str) -> str:
    # fallback for non-pdf files: first title-ish line that isn't a header/notice
    if pages:
        for line in pages[0].text.splitlines()[:15]:
            line = line.strip().lstrip("#").strip()
            if _looks_like_title(line) and len(line.split()) >= 3 and not _BOILERPLATE.search(line):
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
