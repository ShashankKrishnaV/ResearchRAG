import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    # tiny .env reader so we don't need python-dotenv
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))

    embed_model: str = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_reranker: bool = os.getenv("RAG_USE_RERANKER", "1") == "1"

    llm_model: str = os.getenv("RAG_LLM_MODEL", "command-r7b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm_temperature: float = float(os.getenv("RAG_LLM_TEMPERATURE", "0.2"))

    chunk_words: int = int(os.getenv("RAG_CHUNK_WORDS", "220"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "40"))

    candidates: int = int(os.getenv("RAG_CANDIDATES", "30"))  # per retriever, before rerank
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


settings = Settings()
