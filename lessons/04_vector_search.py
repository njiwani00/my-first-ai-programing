"""
LESSON 4 - VECTOR SEARCH (RETRIEVAL)

Run:  python lessons/04_vector_search.py

Now we connect lessons 2 and 3:
    every chunk -> embedding -> stored in a VectorStore
    question    -> embedding -> compare with ALL stored vectors -> top k

This is the "R" in RAG. Open rag/vector_store.py: the search is one line,
    scores = self.vectors @ query_vector
"""

from _common import DATA_DIR, PROJECT_ROOT, pause, short, title

from rag import Config, VectorStore, chunk_documents, get_embedder, load_documents

title("LESSON 4: Vector search")
config = Config()
print("Settings:", config.summary())

# ---- Build the index (this happens ONCE) -----------------------------------
documents = load_documents(DATA_DIR)
chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
embedder = get_embedder(config.embedder, config.st_model, config)
embedder.fit([c.text for c in chunks])
vectors = embedder.embed([c.text for c in chunks])

store = VectorStore()
store.add(chunks, vectors)
print(f"\nIndex built: {len(store)} chunks, each a vector of {vectors.shape[1]} numbers.")
pause()

# ---- Search (this happens for EVERY question) -------------------------------
questions = [
    "When does the library close on Saturday?",
    "What is the passing mark?",
    "Why is Mars red?",
    "What is a dictionary in Python?",
    "Who won the cricket world cup?",  # not in our documents!
]
for q in questions:
    title(f"Q: {q}")
    results = store.search(embedder.embed([q])[0], top_k=3, min_score=config.min_score)
    if not results:
        print("   No chunk passed the minimum score -> nothing relevant in the documents.")
    for rank, r in enumerate(results, 1):
        print(f"  #{rank} score={r.score:.3f}  [{r.chunk.source}]")
        print(f"      {short(r.chunk.text, 150)}")
pause()

# ---- Save and load the index --------------------------------------------------
title("Saving the index")
index_dir = PROJECT_ROOT / "index"
store.save(index_dir)
loaded = VectorStore.load(index_dir)
print(f"Saved to index/ and loaded back: {len(loaded)} chunks.")
print("Real vector databases (Chroma, FAISS, Qdrant...) do this at a huge scale.")

title("Try it yourself")
print(
    "1. Change top_k to 1 and to 5. When is more context helpful? When is it noise?\n"
    "2. Ask your own question by editing the `questions` list.\n"
    "3. What does the score look like for a question with NO answer in the documents?"
)
