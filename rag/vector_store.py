"""
STEP 4 - STORE & SEARCH: keep vectors and find the closest ones to a query.

This is a tiny "vector database" written with numpy.
Real projects use Chroma, FAISS, Qdrant, pgvector, Pinecone... but they all
do the same core job you can read below in ~40 lines:

    similarity = (all chunk vectors) . (query vector)
    return the top_k chunks with the highest similarity
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np

from .chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity: 1.0 = identical direction, 0 = unrelated


class VectorStore:
    def __init__(self):
        self.vectors = None  # shape: (number_of_chunks, vector_size)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Need exactly one vector per chunk")
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3, min_score: float = 0.0) -> List[SearchResult]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
        query_vector = np.asarray(query_vector).reshape(-1)
        # Vectors are normalised, so the dot product IS the cosine similarity.
        scores = self.vectors @ query_vector
        best = np.argsort(-scores)[:top_k]
        return [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in best
            if scores[i] >= min_score
        ]

    # ---- saving / loading (so you don't re-embed every time) ----
    def save(self, folder: str) -> None:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, folder: str) -> "VectorStore":
        path = Path(folder)
        store = cls()
        store.vectors = np.load(path / "vectors.npy")
        with open(path / "chunks.json", encoding="utf-8") as f:
            store.chunks = [Chunk(**c) for c in json.load(f)]
        return store
