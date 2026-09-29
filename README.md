# ResearchRAG

A small, fully local Retrieval-Augmented Generation app for research papers.
Upload your papers on one page, ask questions on another, and get answers grounded
in *your* library — every claim linked back to the paper and page it came from.

No paid APIs. Embeddings, reranking and generation all run on your machine.

![Architecture](docs/architecture.svg)

## How it works

**Ingest (Library page)**

1. You drop in PDFs, DOCX, TXT or Markdown files.
2. Text is extracted page by page (so citations can point to a page) and the paper title is detected.
3. Pages are split into ~220-word, sentence-aware chunks with a small overlap.
4. Each chunk gets a dense embedding (`BAAI/bge-small-en-v1.5`) and BM25 keyword terms.
5. Everything is persisted as plain files under `data/` — no database to run.

**Ask (Ask page)**

1. Your question is searched two ways: dense (semantic) and BM25 (exact terms, acronyms, formulas).
2. The two rankings are merged with Reciprocal Rank Fusion.
3. A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rereads the top candidates against the question.
   Only passages it scores **≥ 30% relevant** are kept (up to 6). If none qualify, the LLM isn't called at all —
   you get a "nothing relevant found" message instead of a stretched answer.
4. Those chunks go to **Cohere Command R7B** running locally in Ollama, which answers using only the sources and cites them as `[1]`, `[2]`…
5. The UI streams the answer and shows a source card per citation: paper, page, relevance, snippet, and a link that opens the PDF at that page.

If Ollama isn't running you still get the reranked passages with citations — retrieval never depends on the LLM.

## Why these choices

| Piece | Choice | Reason |
|---|---|---|
| Retrieval | Hybrid dense + BM25 | Dense handles paraphrase, BM25 catches exact terms like "BERT-large" or "Eq. 3". |
| Fusion | RRF (k=60) | Rank-based, so no score calibration between the two retrievers. |
| Rerank | Cross-encoder | Scores query and chunk together — much sharper than bi-encoder similarity. |
| Cutoff | Reranker probability ≥ 0.30 | Cosine scores from bge are compressed (unrelated text often scores 0.6+), so the calibrated reranker score is what gets thresholded. |
| LLM | Command R7B (Ollama) | 7B model trained for grounded RAG with citations, runs on a laptop. |
| Storage | `.npy` + `.jsonl` files | Transparent, portable, fast enough for thousands of papers. |
| API/UI | FastAPI + vanilla JS | Two clean pages, zero build step. |

## Project layout

```
ResearchRAG/
├── app/
│   ├── main.py        # FastAPI routes + static pages
│   ├── config.py      # settings (env overridable)
│   ├── ingest.py      # parsing + chunking
│   ├── models.py      # embedder + reranker (lazy loaded)
│   ├── store.py       # on-disk index, doc registry
│   ├── bm25.py        # small BM25 implementation
│   ├── retriever.py   # hybrid search, RRF, rerank
│   └── llm.py         # prompt + Ollama streaming client
├── web/               # Library + Ask pages (HTML/CSS/JS)
├── tests/             # pipeline tests (no model downloads needed)
├── data/              # created at runtime: uploads/ and index/
├── docs/architecture.svg
├── requirements.txt
└── run.sh
```

## Setup

Requirements: Python 3.9+, [Ollama](https://ollama.com), ~6 GB free disk for the models.

```bash
# 1. local LLM
ollama pull command-r7b

# 2. python deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. run
./run.sh            # or: uvicorn app.main:app --reload
```

Open http://localhost:8000 — **Library** to upload, **Ask** to query.

The embedding and reranker models (~200 MB total) download automatically on first start.

## Configuration

All settings live in `app/config.py` and can be overridden with environment variables
(copy `.env.example` to `.env`):

| Variable | Default |
|---|---|
| `RAG_DATA_DIR` | `./data` |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` |
| `RAG_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RAG_LLM_MODEL` | `command-r7b` |
| `OLLAMA_URL` | `http://localhost:11434` |
| `RAG_CHUNK_WORDS` / `RAG_CHUNK_OVERLAP` | `220` / `40` |
| `RAG_TOP_K` | `6` (max sources per answer) |
| `RAG_MIN_RELEVANCE` | `0.30` (reranker cutoff, adjustable per question in the UI) |

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | upload one or more files |
| `GET` | `/api/documents` | list indexed papers |
| `DELETE` | `/api/documents/{id}` | remove a paper and its chunks |
| `GET` | `/api/documents/{id}/file` | original file (append `#page=N` for PDFs) |
| `POST` | `/api/search` | retrieval only → ranked, cited chunks |
| `POST` | `/api/ask` | streamed answer (NDJSON: `sources`, `token`, `done`) |
| `GET` | `/api/health` | index stats + LLM availability |

## Tests

```bash
pytest -q
```

Tests use a tiny hashing embedder, so they run offline in seconds.

## Roadmap

- Scanned PDF support via OCR
- Per-paper summaries and "compare these two papers"
- Export answers with a formatted bibliography (BibTeX / APA)
