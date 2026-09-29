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
    # the title is the biggest horizontal text on page 1 — but judged per *line*,
    # so small caps, inline math or a mixed-size title don't get cut into pieces
    lines = []   # each: {"y", "size", "text"}

    def visit(text, cm, tm, font_dict, font_size):
        rotated = any(abs(m) > 0.01 for m in (tm[1], tm[2], cm[1], cm[2]))
        if not text or rotated:   # skip rotated text, e.g. the arXiv side stamp
            return
        if not text.strip():
            # bare "\n"/spaces arrive with a placeholder position; keep the break, don't open a line
            if lines:
                lines[-1]["text"] += "\n" if "\n" in text else " "
            return
        size = round(font_size * (abs(tm[3] * cm[3]) or 1.0), 1)
        y = tm[5] * cm[3] + cm[5]
        has_letters = any(c.isalpha() for c in text)
        cur = lines[-1] if lines else None

        if cur is not None and tm[4] == 0 and tm[5] == 0:
            # pypdf sometimes reports a placeholder origin for text right after a line break;
            # treat it as the next line down (or the same line if no break happened)
            y = cur["y"] - 1.2 * size if cur["text"].endswith("\n") else cur["y"]

        if cur is None or abs(y - cur["y"]) > 0.5 * max(size, cur["size"], 1.0):
            cur = {"y": y, "size": 0.0, "text": ""}
            lines.append(cur)
        cur["text"] += text
        if has_letters:
            cur["size"] = max(cur["size"], size)

    try:
        page.extract_text(visitor_text=visit)
    except Exception:
        return None

    for ln in lines:
        ln["text"] = " ".join(ln["text"].split())
    ok = [i for i, ln in enumerate(lines)
          if sum(c.isalpha() for c in ln["text"]) >= 2 and not _BOILERPLATE.search(ln["text"])]
    if not ok:
        return None

    top = max(lines[i]["size"] for i in ok)
    anchor = next(i for i in ok if lines[i]["size"] >= top - 0.6)

    def belongs(i, ref):
        # a neighbouring line is part of the title if it's nearly as large and close by
        ln = lines[i]
        return (i in ok and ln["size"] >= 0.8 * top
                and abs(ln["y"] - lines[ref]["y"]) <= 2.2 * top)

    start = end = anchor
    while start - 1 >= 0 and end - start < 3 and belongs(start - 1, start):
        start -= 1
    while end + 1 < len(lines) and end - start < 3 and belongs(end + 1, end):
        end += 1

    title = " ".join(lines[i]["text"] for i in range(start, end + 1)).strip(" *†‡∗")
    return title if _looks_like_title(title) else None


def _usable_metadata_title(t: str) -> bool:
    # metadata is often junk like "arXiv preprint" or "Conference Paper" — same filter as page text
    return (_looks_like_title(t)
            and not _BOILERPLATE.search(t)
            and not re.search(r"\.(dvi|pdf|tex|docx?)$|^microsoft word|^untitled", t, re.I))


def _read_pdf(path: Path) -> tuple[list[Page], str | None]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [Page(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]

    title = None
    try:
        t = str((reader.metadata or {}).get("/Title") or "").strip()
        if _usable_metadata_title(t):
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
