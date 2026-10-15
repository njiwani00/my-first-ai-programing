"""
LESSON 3 - EMBEDDINGS: TURNING TEXT INTO NUMBERS

Run:  python lessons/03_embeddings.py

Computers can't compare meanings directly, but they CAN compare numbers.
An embedding turns a piece of text into a vector (a list of numbers).
Similar texts -> vectors pointing in similar directions.

We measure "pointing in a similar direction" with COSINE SIMILARITY:
     cos(a, b) = (a . b) / (|a| * |b|)
     1.0 = same direction (very similar), 0.0 = nothing in common

In this lesson we:
  Part A - build the simplest embedding BY HAND (word counts)
  Part B - use TF-IDF (what this project uses by default)
  Part C - see where word-based embeddings fail (and why neural ones exist)
  Part D - (if EMBEDDER=openai in .env) the same test with OpenAI embeddings
"""

import numpy as np
from _common import pause, title

title("LESSON 3: Embeddings")

# ============================================================================
# Part A: a "bag of words" embedding written from scratch
# ============================================================================
sentences = [
    "the library closes at ten",
    "the library opens at eight",
    "venus is the hottest planet",
]
vocabulary = sorted({word for s in sentences for word in s.split()})
print("Vocabulary:", vocabulary)


def bag_of_words(text):
    """Count how many times each vocabulary word appears."""
    words = text.split()
    return np.array([words.count(v) for v in vocabulary], dtype=float)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


vectors = [bag_of_words(s) for s in sentences]
print("\nEach sentence as a vector (one number per vocabulary word):")
for s, v in zip(sentences, vectors):
    print(f"  {s:30s} -> {v.astype(int).tolist()}")

query = "when does the library close"
qv = bag_of_words(query)
print(f"\nQuery: '{query}'")
for s, v in zip(sentences, vectors):
    print(f"  similarity to '{s}': {cosine(qv, v):.2f}")
print("\nThe library sentences score higher than the Venus sentence. That's retrieval!")
print("But 'close' vs 'closes' don't match... word-based methods are picky.")
pause()

# ============================================================================
# Part B: TF-IDF - smarter word counting
# ============================================================================
title("Part B: TF-IDF")
print(
    "TF-IDF = Term Frequency x Inverse Document Frequency.\n"
    "Words that appear in EVERY text (like 'the') get a low weight.\n"
    "Rare, specific words (like 'venus', 'library') get a high weight."
)
from rag.embedder import TfidfEmbedder  # noqa: E402

texts = [
    "Late submissions lose 10 percent of the marks for each day.",
    "The library fine is 10 rupees per day for a late book.",
    "Venus is the hottest planet because of its thick atmosphere.",
    "Chlorophyll absorbs red and blue light in the leaves.",
]
embedder = TfidfEmbedder()
embedder.fit(texts)
matrix = embedder.embed(texts)
print(f"\n{len(texts)} texts -> matrix of shape {matrix.shape} (texts x vocabulary words)")

for q in ["what happens if I submit my assignment late", "which light do leaves absorb", "who invented the telephone"]:
    qv = embedder.embed([q])[0]
    scores = matrix @ qv  # vectors are normalised, so dot product = cosine
    best = int(np.argmax(scores))
    print(f"\nQuery: {q}")
    for t, s in zip(texts, scores):
        marker = "  <-- best" if t == texts[best] and s > 0 else ""
        print(f"   {s:.2f}  {t}{marker}")
print("\n'who invented the telephone' shares no words with any text -> every score is 0.")
pause()

# ============================================================================
# Part C: the weakness of word matching
# ============================================================================
title("Part C: Where word matching fails")
q = "Can I use ChatGPT to help write my essay?"
target = "AI tools such as chatbots may be used for brainstorming and for checking grammar."
emb = TfidfEmbedder()
emb.fit([target, q])
score = float(emb.embed([q])[0] @ emb.embed([target])[0])
print(f"Query : {q}\nText  : {target}\nTF-IDF similarity: {score:.2f}")
print(
    "\nA human sees these are about the same thing, but they share almost no words!\n"
    "NEURAL embeddings (e.g. sentence-transformers) learn MEANING, so 'ChatGPT'\n"
    "lands close to 'AI tools' and 'chatbots'.\n\n"
    "Try it: pip install sentence-transformers, then set EMBEDDER=sentence-transformers\n"
    "in your .env file and run lesson 4 and the app again."
)

# ============================================================================
# Part D: the same test with OpenAI embeddings (only if EMBEDDER=openai)
# ============================================================================
from rag import Config  # noqa: E402

if Config().embedder == "openai":
    pause()
    title("Part D: OpenAI embeddings understand MEANING")
    from rag.embedder import get_embedder  # noqa: E402

    try:
        cfg = Config()
        oe = get_embedder("openai", config=cfg)
        unrelated = "Venus is the hottest planet because of its thick atmosphere."
        q_vec, t_vec, u_vec = oe.embed([q, target, unrelated])
        print(f"Query    : {q}")
        print(f"Related  : {target}\n           similarity {float(q_vec @ t_vec):.2f}")
        print(f"Unrelated: {unrelated}\n           similarity {float(q_vec @ u_vec):.2f}")
        print(f"\nEach text became a vector of {len(q_vec)} numbers.")
        if float(q_vec @ t_vec) > float(q_vec @ u_vec):
            print("The related sentence scores higher, even though they share almost no words!")
            print("Compare with Part C, where TF-IDF gave it 0.00.")
        else:
            print("Compare these scores with Part C. Did meaning-based embeddings help?")
    except RuntimeError as err:
        print(err)
