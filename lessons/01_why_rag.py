"""
LESSON 1 - WHY DO WE NEED RAG?

Run:  python lessons/01_why_rag.py

Big idea
--------
A language model (LLM) only knows what was in its training data.
It does NOT know:
  * your college's rules, your company's documents, your class notes
  * anything that happened after it was trained
When it doesn't know, it may confidently make something up ("hallucinate").

RAG = Retrieval-Augmented Generation
  1. RETRIEVE  the most relevant pieces of YOUR documents
  2. AUGMENT   the question by pasting those pieces into the prompt
  3. GENERATE  an answer with the LLM, which now has the facts in front of it

It's an OPEN-BOOK exam for the model, instead of a closed-book one.
"""

from _common import DATA_DIR, pause, title

title("LESSON 1: Why RAG?")

question = "What is the fine for returning a library book late at Greenfield College?"
print(f"\nStudent asks: {question}")

# ---- A "closed-book" model: it only has its training memory -------------
title("Without RAG (closed-book)")
print(
    "The model has never seen Greenfield College's handbook (it's fictional!).\n"
    "Typical closed-book answers are guesses like:\n"
    '   "Library fines are usually around 1 dollar per day..."\n'
    "That sounds confident but it's invented -> a HALLUCINATION."
)
pause()

# ---- An "open-book" approach: look it up first ---------------------------
title("With RAG (open-book)")
handbook = (DATA_DIR / "greenfield_college_handbook.md").read_text(encoding="utf-8")

# The simplest possible retrieval: find lines that contain our keywords.
keywords = ["fine", "late book"]
found = [line for line in handbook.splitlines() if all(k in line.lower() for k in keywords)]

print("Step 1 - RETRIEVE: lines from the handbook that match the question:")
for line in found:
    print("   >", line)

print("\nStep 2 - AUGMENT: put that text into the prompt:")
prompt = f"Context:\n{found[0] if found else ''}\n\nQuestion: {question}\nAnswer using only the context."
print("-" * 70)
print(prompt)
print("-" * 70)

print("\nStep 3 - GENERATE: the model reads the context and answers:")
print("   The fine is 10 rupees per day, up to a maximum of 500 rupees per book.")
pause()

title("What you learned")
print(
    "* LLMs can't know private or new information -> they hallucinate.\n"
    "* RAG retrieves relevant text and gives it to the model as context.\n"
    "* Keyword matching (what we did here) breaks easily: try asking\n"
    "  'How much do I pay if my book is overdue?' - no line contains 'fine'!\n"
    "  Lessons 2-4 build a much better retriever."
)
