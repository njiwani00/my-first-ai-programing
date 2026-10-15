"""
rag - a small, readable Retrieval-Augmented Generation library for learning.

Every RAG system has the same five steps. Each step lives in its own file:

    1. loader.py        -> read documents from disk (.txt, .md, .pdf)
    2. chunker.py       -> split long documents into small overlapping chunks
    3. embedder.py      -> turn text into vectors (lists of numbers)
    4. vector_store.py  -> store vectors and find the most similar ones
    5. generator.py     -> ask a language model to answer using the found chunks

pipeline.py glues all five steps together into one easy-to-use class.
"""

from .config import Config
from .loader import Document, load_documents
from .chunker import Chunk, chunk_documents, chunk_text
from .embedder import get_embedder
from .vector_store import VectorStore, SearchResult
from .generator import build_prompt, get_generator
from .pipeline import RAGPipeline, Answer

__all__ = [
    "Config",
    "Document",
    "load_documents",
    "Chunk",
    "chunk_text",
    "chunk_documents",
    "get_embedder",
    "VectorStore",
    "SearchResult",
    "build_prompt",
    "get_generator",
    "RAGPipeline",
    "Answer",
]
