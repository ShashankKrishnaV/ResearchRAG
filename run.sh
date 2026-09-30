#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "→ creating virtualenv"
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

# .env is gitignored, so fresh clones start from the committed template
if [ ! -f .env ] && [ -f .env.example ]; then
  echo "→ creating .env from .env.example"
  cp .env.example .env
fi

# same precedence as app/config.py: shell variable > .env > default
env_model="$(grep -E '^RAG_LLM_MODEL=' .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d "\"' ")"
MODEL="${RAG_LLM_MODEL:-${env_model:-command-r7b}}"
if command -v ollama >/dev/null 2>&1; then
  if ! curl -s http://localhost:11434/api/tags >/dev/null; then
    echo "→ starting ollama"
    (ollama serve >/dev/null 2>&1 &) ; sleep 2
  fi
  ollama list | grep -q "^$MODEL" || { echo "→ pulling $MODEL"; ollama pull "$MODEL"; }
else
  echo "! ollama not found — retrieval works, answers need https://ollama.com"
fi

echo "→ http://localhost:8000"
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"
