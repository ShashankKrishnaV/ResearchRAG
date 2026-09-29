"""Grounded answer generation with Command R7B served by Ollama."""
from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from .config import settings

SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question using ONLY the numbered sources provided.

Rules:
- Cite every factual sentence with the source number in square brackets, e.g. [2] or [1][3].
- Never cite a number that isn't in the sources, and never invent facts, numbers or paper names.
- If the sources don't contain the answer, say so plainly and mention what they do cover.
- When sources disagree, point out the disagreement and cite both sides.
- Be concise and well structured: a direct answer first, then supporting detail. Use short bullet lists where it helps."""


class LLMUnavailable(RuntimeError):
    pass


def format_sources(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        where = f", page {h['page']}" if h.get("page") else ""
        blocks.append(f"[{h['ref']}] {h['title']}{where}\n{h['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    user = f"Sources:\n\n{format_sources(hits)}\n\n---\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def stream_answer(question: str, hits: list[dict]) -> Iterator[str]:
    body = {
        "model": settings.llm_model,
        "messages": build_messages(question, hits),
        "stream": True,
        "options": {"temperature": settings.llm_temperature, "num_ctx": 8192},
    }
    try:
        with httpx.stream("POST", f"{settings.ollama_url}/api/chat", json=body, timeout=httpx.Timeout(300, connect=5)) as r:
            if r.status_code != 200:
                r.read()
                raise LLMUnavailable(f"Ollama returned {r.status_code}: {r.text[:200]}")
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise LLMUnavailable(msg["error"])
                token = msg.get("message", {}).get("content", "")
                if token:
                    yield token
                if msg.get("done"):
                    break
    except httpx.HTTPError as e:
        raise LLMUnavailable(f"Can't reach Ollama at {settings.ollama_url} ({e.__class__.__name__}). Is `ollama serve` running?") from e


def llm_status() -> dict:
    try:
        r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=2)
        names = [m.get("name", "") for m in r.json().get("models", [])]
        ready = any(n.split(":")[0] == settings.llm_model.split(":")[0] for n in names)
        return {"reachable": True, "model": settings.llm_model, "model_ready": ready}
    except Exception:
        return {"reachable": False, "model": settings.llm_model, "model_ready": False}
