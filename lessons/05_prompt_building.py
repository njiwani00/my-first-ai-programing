"""
LESSON 5 - AUGMENTING THE PROMPT

Run:  python lessons/05_prompt_building.py

The "A" in RAG. We build a prompt that contains:
  1. instructions (a system prompt) - "answer ONLY from the context, cite sources,
     say 'I don't know' if the answer isn't there"
  2. the retrieved chunks, numbered [1], [2], [3] with their source file
  3. the user's question

Good instructions are what stop the model from hallucinating.
Open rag/generator.py to see SYSTEM_PROMPT, HYBRID_SYSTEM_PROMPT and build_prompt().

The ONLY difference between "strict" and "hybrid" mode (ANSWER_MODE in .env)
is the instruction text - the retrieval is identical. Prompts are powerful!
"""

from _common import pause, title

from rag import RAGPipeline, build_prompt
from rag.generator import HYBRID_SYSTEM_PROMPT, SYSTEM_PROMPT

title("LESSON 5: Building the augmented prompt")

rag = RAGPipeline()
rag.ingest()

question = "Can I submit my assignment two days late?"
results = rag.retrieve(question)

print("SYSTEM PROMPT for STRICT mode (documents only):")
print("-" * 70)
print(SYSTEM_PROMPT)
print("-" * 70)
print("\nSYSTEM PROMPT for HYBRID mode (documents first, then general knowledge):")
print("-" * 70)
print(HYBRID_SYSTEM_PROMPT)
print("-" * 70)
print(f"\nYour .env currently uses: ANSWER_MODE={rag.config.answer_mode}")
pause()

print("\nUSER PROMPT (context + question):")
print("-" * 70)
print(build_prompt(question, results))
print("-" * 70)

title("Discuss")
print(
    "* Why do we number the passages?          -> so the answer can cite [1], [2]\n"
    "* Why include the source file name?        -> so users can check the original\n"
    "* Why 'say I don't know'?                  -> to reduce hallucinations\n"
    "* Why label 'From general knowledge:'?     -> so users know it is NOT from the documents\n"
    "* What if we retrieved the WRONG chunks?   -> the answer will be wrong too!\n"
    "  Retrieval quality is usually the most important part of a RAG system."
)
