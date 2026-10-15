"""
LESSON 6 - GENERATING THE ANSWER

Run:  python lessons/06_generation.py

The "G" in RAG. The provider is chosen in the .env file (LLM_PROVIDER):

  extractive (default)  - no LLM; copies the best-matching sentences.
                          Free and offline, but can't rephrase or reason.
  ollama                - a real LLM running free on your own laptop
  anthropic / openai    - powerful cloud LLMs (need an API key)

Run this lesson once with "extractive", then with a real LLM, and compare!
"""

from _common import pause, title

from rag import Config, RAGPipeline

config = Config()
title(f"LESSON 6: Generation with provider = '{config.llm_provider}', mode = '{config.answer_mode}'")

rag = RAGPipeline(config)
rag.ingest()

questions = [
    "Can I submit my assignment two days late? How many marks would I lose?",
    "What grade is given for a score of 85?",
    "Why do leaves look green?",
    "What is the capital of Australia?",  # not in the documents
]

for q in questions:
    try:
        answer = rag.ask(q)
    except RuntimeError as err:
        print(f"\nProblem with provider '{config.llm_provider}':\n{err}")
        print("Tip: set LLM_PROVIDER=extractive in .env to run without an API key.")
        break
    print(f"\nQ: {q}")
    print(f"A: {answer.text}")
    if answer.used_documents:
        for i, r in enumerate(answer.sources, 1):
            print(f"   [{i}] {r.chunk.source} (score {r.score:.2f})")
    else:
        print("   source: general knowledge (no document used)")
    print(f"   time: {answer.timings}")
    pause()

title("Discuss")
print(
    "* The extractive answerer can only COPY sentences. For 'two days late, how many\n"
    "  marks?' it can't calculate 2 x 10% = 20%. A real LLM can reason over the context.\n"
    "* 'Capital of Australia' isn't in our documents.\n"
    "    ANSWER_MODE=strict -> 'I don't know' (safe: never invents, only trusts the documents)\n"
    "    ANSWER_MODE=hybrid -> 'From general knowledge: Canberra' (helpful, clearly labelled)\n"
    "  Change ANSWER_MODE in .env, run again and compare. Which would you use for a\n"
    "  college help desk? For a homework helper? Discuss the trade-off!"
)
