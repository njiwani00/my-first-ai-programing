"""
Configuration: every "knob" of the RAG system in one place.

Values come from (in order of priority):
    1. arguments you pass in code, e.g. Config(top_k=5)
    2. environment variables / the .env file
    3. the defaults written below
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    # python-dotenv reads the .env file in the project folder (if it exists)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Config:
    # Where the documents live
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", str(PROJECT_ROOT / "data")))

    # Chunking: how big each piece of text is, and how much neighbours overlap
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "100")))

    # Embeddings: "tfidf" (offline), "openai" (needs key) or "sentence-transformers"
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "tfidf"))
    st_model: str = field(default_factory=lambda: _env("ST_MODEL", "all-MiniLM-L6-v2"))
    openai_embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Retrieval: how many chunks to fetch and the minimum similarity to keep one
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "3")))
    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0.05")))

    # Answer mode: "strict" = documents only, "hybrid" = documents first, then general knowledge
    answer_mode: str = field(default_factory=lambda: _env("ANSWER_MODE", "strict").lower().strip())

    # Generation: "extractive" (no API key), "anthropic", "openai" or "ollama"
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "extractive"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.2")))
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "500")))

    def summary(self) -> str:
        return (
            f"data_dir={self.data_dir} | chunk_size={self.chunk_size} overlap={self.chunk_overlap} | "
            f"embedder={self.embedder} | top_k={self.top_k} | llm={self.llm_provider} | mode={self.answer_mode}"
        )
