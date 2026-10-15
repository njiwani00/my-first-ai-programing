"""
The full RAG pipeline in one class.

    rag = RAGPipeline()          # uses settings from .env
    rag.ingest("data")           # load -> chunk -> embed -> store
    answer = rag.ask("What is the late submission policy?")
    print(answer.text)           # retrieve -> augment -> generate
    for r in answer.sources: print(r.chunk.source, r.score)
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .chunker import chunk_documents
from .config import Config
from .embedder import get_embedder
from .generator import build_prompt, get_generator, is_general_knowledge
from .loader import load_documents
from .vector_store import SearchResult, VectorStore


@dataclass
class Answer:
    question: str
    text: str
    sources: List[SearchResult]
    prompt: str
    timings: dict = field(default_factory=dict)  # seconds spent in each step
    used_documents: bool = True  # False when the model answered from general knowledge (hybrid mode)


class RAGPipeline:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.embedder = get_embedder(self.config.embedder, self.config.st_model, self.config)
        self.store = VectorStore()
        self._generator = None  # created lazily, so retrieval works without an API key

    @property
    def generator(self):
        if self._generator is None:
            self._generator = get_generator(self.config)
        return self._generator

    # ---------------- INDEXING (done once) ----------------
    def ingest(self, folder: Optional[str] = None, verbose: bool = False) -> int:
        folder = folder or self.config.data_dir
        documents = load_documents(folder)
        if not documents:
            raise ValueError(f"No .txt/.md/.pdf files found in {folder}")
        chunks = chunk_documents(documents, self.config.chunk_size, self.config.chunk_overlap)

        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)

        self.store = VectorStore()
        self.store.add(chunks, vectors)
        if verbose:
            print(f"Indexed {len(documents)} documents -> {len(chunks)} chunks "
                  f"(vector size {vectors.shape[1]})")
        return len(chunks)

    # ---------------- QUERYING (every question) ----------------
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[SearchResult]:
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, top_k or self.config.top_k, self.config.min_score)

    def ask(self, question: str, top_k: Optional[int] = None) -> Answer:
        if len(self.store) == 0:
            raise RuntimeError("The index is empty - call ingest() first.")

        t0 = time.perf_counter()
        results = self.retrieve(question, top_k)
        t1 = time.perf_counter()
        prompt = build_prompt(question, results)
        text = self.generator.generate(question, results)
        t2 = time.perf_counter()

        return Answer(
            question=question,
            text=text,
            sources=results,
            prompt=prompt,
            timings={"retrieve": round(t1 - t0, 3), "generate": round(t2 - t1, 3)},
            used_documents=bool(results) and not is_general_knowledge(text),
        )
