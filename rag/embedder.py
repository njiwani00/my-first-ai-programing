"""
STEP 3 - EMBED: turn text into vectors (lists of numbers).

Texts with similar meaning should get vectors that point in similar directions.
Then "find relevant text" becomes "find vectors with high cosine similarity".

Three embedders are provided (choose with EMBEDDER in .env):

  TfidfEmbedder  (EMBEDDER=tfidf)
      Counts words, giving rare words more weight (TF-IDF).
      + Works offline, instant, no downloads.
      - Matches WORDS, not meaning: "car" and "automobile" look unrelated.

  OpenAIEmbedder  (EMBEDDER=openai, needs OPENAI_API_KEY)
      OpenAI's embedding model, trained on huge amounts of text.
      + Understands meaning: "ChatGPT" lands close to "AI tools".
      - Needs internet and costs a tiny amount per text.
        Results are cached in the index/ folder, so each text is paid for once.

  SentenceTransformerEmbedder  (EMBEDDER=sentence-transformers)
      A small neural network trained to put similar MEANINGS close together.
      + Understands synonyms and paraphrases.
      - Downloads a ~90 MB model the first time.

All return L2-normalised vectors, so cosine similarity = dot product.
"""

import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np

from . import api


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

        # Stop words are common words ("the", "is", "does") that carry no topic.
        # sklearn's list misses a few question words, so we add them.
        stop_words = list(ENGLISH_STOP_WORDS | {"does", "did", "doing", "tell", "explain"})
        # TRY IT: ngram_range=(1, 2) also counts word pairs like "late submission".
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 1), sublinear_tf=True)
        self.fitted = False

    def fit(self, texts: List[str]) -> None:
        """TF-IDF must first learn the vocabulary from our documents."""
        self.vectorizer.fit(texts)
        self.fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() on your documents before embed().")
        vectors = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return normalize(vectors)


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "This embedder needs:  pip install sentence-transformers\n"
                "Or set EMBEDDER=tfidf in your .env file."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> None:
        pass  # pre-trained: nothing to learn

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return normalize(vectors.astype(np.float32))


class OpenAIEmbedder:
    name = "openai"
    BATCH = 100  # texts per API request

    def __init__(self, model: str = "text-embedding-3-small", base_url: str = "https://api.openai.com/v1",
                 cache_file: str = None):
        self.api_key = api.openai_key()
        self.model = model
        self.url = base_url.rstrip("/") + "/embeddings"
        project_root = Path(__file__).resolve().parent.parent
        self.cache_file = Path(cache_file) if cache_file else project_root / "index" / "openai_embedding_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self.cache), encoding="utf-8")

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}::{text}".encode("utf-8")).hexdigest()

    def fit(self, texts: List[str]) -> None:
        pass  # pre-trained: nothing to learn

    def embed(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self.cache]
        for start in range(0, len(missing), self.BATCH):
            batch = missing[start : start + self.BATCH]
            data = api.post_json(self.url, {"model": self.model, "input": batch},
                                 {"Authorization": f"Bearer {self.api_key}"})
            for item in data["data"]:
                self.cache[self._key(batch[item["index"]])] = item["embedding"]
        if missing:
            self._save_cache()
        vectors = np.array([self.cache[self._key(t)] for t in texts], dtype=np.float32)
        return normalize(vectors)


def get_embedder(kind: str = "tfidf", st_model: str = "all-MiniLM-L6-v2", config=None):
    kind = kind.lower().strip()
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind == "openai":
        if config is not None:
            return OpenAIEmbedder(config.openai_embedding_model, config.openai_base_url)
        return OpenAIEmbedder()
    if kind in ("sentence-transformers", "st", "sbert"):
        return SentenceTransformerEmbedder(st_model)
    raise ValueError(f"Unknown embedder '{kind}'. Use 'tfidf', 'openai' or 'sentence-transformers'.")
