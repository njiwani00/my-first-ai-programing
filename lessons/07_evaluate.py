"""
LESSON 7 - THE FULL PIPELINE + EVALUATION

Run:  python lessons/07_evaluate.py

How do we know if our RAG system is good? We TEST it.
evaluation/questions.json has questions with the file that SHOULD be retrieved
and a few words that SHOULD appear in the answer.

Questions are marked "easy" (uses the same words as the documents) or "hard"
(uses different words - e.g. "overdue" instead of "late", "ChatGPT" instead of
"AI tools"). Hard questions show the limits of word-based (TF-IDF) search.
Questions marked "impossible" need a calculation over a whole table (average,
highest). RAG only sees a few chunks, so it is EXPECTED to fail them - see lesson 8.

We measure:
  * Hit rate @k   - % of questions where the correct file is in the top k chunks
  * Answer score  - % of expected words that appear in the answer

Then we change ONE setting at a time (chunk size, top_k) and compare.
This is exactly how engineers improve real RAG systems.
"""

import json

from _common import PROJECT_ROOT, title

from rag import Config, RAGPipeline

QUESTIONS = json.loads((PROJECT_ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))


def evaluate(config: Config, verbose: bool = False) -> dict:
    rag = RAGPipeline(config)
    rag.ingest()
    hits, answer_scores = 0, []
    for item in QUESTIONS:
        answer = rag.ask(item["question"])
        sources = [r.chunk.source for r in answer.sources]
        hit = item["expected_source"] in sources
        hits += hit
        words = item["expected_words"]
        found = sum(w.lower() in answer.text.lower() for w in words)
        answer_scores.append(found / len(words))
        if verbose:
            mark = "OK  " if hit else "MISS"
            print(f"[{mark}] ({item.get('level', '?')}) {item['question']}")
            print(f"        answer: {answer.text[:120]}")
    return {
        "hit_rate": hits / len(QUESTIONS),
        "answer_score": sum(answer_scores) / len(answer_scores),
    }


title("LESSON 7: Evaluating the default settings")
base = Config()
print("Settings:", base.summary(), "\n")
result = evaluate(base, verbose=True)
print(f"\nHit rate: {result['hit_rate']:.0%}   Answer score: {result['answer_score']:.0%}")

title("Experiment: change one setting at a time")
# Running the LLM for 9 settings x all questions would be hundreds of paid API
# calls. Search quality (hit rate) doesn't depend on the LLM, so for this grid the
# answers are written by the free extractive answerer.
print("(Answers in this table use the free extractive answerer to save API cost.)\n")
print(f"{'chunk_size':>10} {'top_k':>6} {'hit rate':>9} {'answer':>7}")
for chunk_size in (200, 500, 1000):
    for top_k in (1, 3, 5):
        cfg = Config(chunk_size=chunk_size, chunk_overlap=chunk_size // 5, top_k=top_k, llm_provider="extractive")
        r = evaluate(cfg)
        print(f"{chunk_size:>10} {top_k:>6} {r['hit_rate']:>9.0%} {r['answer_score']:>7.0%}")

title("Discuss")
print(
    "* Which settings worked best? Why might bigger top_k raise the hit rate?\n"
    "* Add 5 questions of your own to evaluation/questions.json and re-run.\n"
    "* Switch EMBEDDER or LLM_PROVIDER in .env and compare the scores."
)
