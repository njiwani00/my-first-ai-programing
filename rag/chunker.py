"""
STEP 2 - CHUNK: split long documents into small, overlapping pieces.

Why?
  * Language models have a limited context window - we can't send whole books.
  * Small chunks are more "focused", so search finds precisely the relevant part.
  * Overlap keeps a sentence that sits on a boundary from being cut in half
    and losing its meaning.

Strategy used here (simple but good):
  1. Split the text into sentences.
  2. Pack sentences into a chunk until it reaches `chunk_size` characters.
  3. Start the next chunk with the last ~`overlap` characters of sentences.
"""

import re
from dataclasses import dataclass
from typing import List

from .loader import Document


@dataclass
class Chunk:
    text: str
    source: str  # which file it came from
    chunk_id: int  # position of the chunk inside that file


# A sentence ends at . ! or ? followed by a space - but NOT after titles and
# abbreviations like "Dr." or "e.g.", otherwise "Dr. Meera Rao" would be cut in two.
_ABBREVIATIONS = r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bProf\.)(?<!\bSt\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\betc\.)"
SENTENCE_END = _ABBREVIATIONS + r"(?<=[.!?])\s+"

# Split at sentence ends, blank lines, or before markdown headings.
_SENTENCE_SPLIT = re.compile(SENTENCE_END + r"|\n\s*\n|\n(?=#)")


def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _join(sentences: List[str]) -> str:
    """Join sentences with spaces, but keep markdown headings on their own line."""
    out = ""
    for s in sentences:
        out += s + ("\n" if s.startswith("#") else " ")
    return out.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split one text into chunks of roughly `chunk_size` characters."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        # A single very long sentence: cut it into hard pieces.
        if len(sentence) > chunk_size:
            if current:
                chunks.append(_join(current))
                current, current_len = [], 0
            step = chunk_size - overlap
            for start in range(0, len(sentence), step):
                chunks.append(sentence[start : start + chunk_size])
            continue

        if current_len + len(sentence) > chunk_size and current:
            chunks.append(_join(current))
            # Keep the tail sentences as overlap for the next chunk.
            tail: List[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current, current_len = tail, tail_len

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(_join(current))
    return chunks


def chunk_documents(documents: List[Document], chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in documents:
        for i, piece in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            all_chunks.append(Chunk(text=piece, source=doc.source, chunk_id=i))
    return all_chunks
