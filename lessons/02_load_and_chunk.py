"""
LESSON 2 - LOADING AND CHUNKING DOCUMENTS

Run:  python lessons/02_load_and_chunk.py

Why chunk?
  * We can't paste every document into the prompt (too long, too expensive).
  * Search works better on small, focused pieces than on whole files.

Two knobs to understand:
  chunk_size  - max characters per chunk
                too small -> chunks lose context ("It is 10 rupees." What is?)
                too big   -> chunks mix many topics and search gets fuzzy
  overlap     - characters repeated between neighbouring chunks, so facts
                on a boundary aren't split in half

Open rag/loader.py and rag/chunker.py to read the real code.
"""

from _common import DATA_DIR, pause, short, title

from rag import chunk_documents, chunk_text, load_documents

title("LESSON 2: Load and chunk")

# ---- 1. Load ---------------------------------------------------------------
documents = load_documents(DATA_DIR)
print(f"Loaded {len(documents)} documents from {DATA_DIR.name}/")
for doc in documents:
    print(f"   {doc.source:35s} {len(doc.text):6d} characters")
pause()

# ---- 2. Chunk one small text so we can SEE what happens ---------------------
title("Chunking a small example (chunk_size=120, overlap=40)")
example = (
    "The library opens at 8 AM. It closes at 10 PM on weekdays. "
    "On Saturdays it closes at 6 PM. It is closed on Sundays. "
    "Students can borrow 5 books. The late fine is 10 rupees per day."
)
for i, piece in enumerate(chunk_text(example, chunk_size=120, overlap=40)):
    print(f"\nchunk {i}: {piece}")
print("\nNotice: the end of one chunk is repeated at the start of the next (overlap).")
pause()

# ---- 3. Chunk all documents with different sizes ---------------------------
title("Effect of chunk_size on the real documents")
for size in (200, 500, 1000):
    chunks = chunk_documents(documents, chunk_size=size, overlap=size // 5)
    avg = sum(len(c.text) for c in chunks) / len(chunks)
    print(f"chunk_size={size:5d} -> {len(chunks):3d} chunks, average length {avg:.0f} chars")
pause()

title("First 3 chunks of the handbook (chunk_size=500)")
chunks = chunk_documents(documents, chunk_size=500, overlap=100)
handbook_chunks = [c for c in chunks if c.source.startswith("greenfield")]
for c in handbook_chunks[:3]:
    print(f"\n[{c.source} #{c.chunk_id}] {short(c.text, 300)}")

title("Try it yourself")
print(
    "1. Change chunk_size to 100 and look at the chunks. Are they still meaningful?\n"
    "2. Put your own .txt, .md or .pdf file into the data/ folder and run again."
)
