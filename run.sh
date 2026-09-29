#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "→ creating virtualenv"
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

MODEL="${RAG_LLM_MODEL:-command-r7b}"
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
