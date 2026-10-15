# Exercises

Work through these after the lessons. Each one changes one small thing so you can see its effect.
Solutions are hints, not code: the point is to try!

## Level 1 - Explore (no coding)

1. **Your own documents.** Put a PDF of your class notes (or any .txt / .md file) into `data/`. Run `python app.py` and ask questions about it. Which questions work well? Which fail?
2. **Chunk size.** In `.env`, set `CHUNK_SIZE=150` and then `CHUNK_SIZE=1500`. Ask "What is the late submission policy?" each time. What changes in the sources and the answer?
3. **top_k.** Run `python app.py --top-k 1` and `python app.py --top-k 8`. When does more context help? When does it add noise?
4. **The prompt.** Run `python app.py --show-prompt`. Find the exact place where the retrieved text is inserted.

## Level 2 - Modify the code

5. **Better TF-IDF.** In `rag/embedder.py`, change `ngram_range=(1, 1)` to `(1, 2)`. Re-run `python lessons/07_evaluate.py`. Did the hit rate change?
6. **Show page numbers.** In `rag/loader.py`, PDFs are joined into one string. Change it so each PDF page becomes its own `Document` with `source="file.pdf (page 3)"`. Now citations show page numbers!
7. **Stricter "I don't know".** In `rag/config.py` increase `min_score`. Find a value where off-topic questions return "I don't know" but real questions still work.
8. **Your own table.** Export a sheet from Excel as CSV (File -> Save As -> CSV UTF-8), put it in `data/`, and ask look-up questions about single rows.
9. **A new file type.** Add support for `.json` or `.html` files in `rag/loader.py`: write a `read_...` function that returns plain text and add the extension to `SUPPORTED`.
10. **A new command.** Add a `:stats` command to `app.py` that prints how many chunks come from each file.

## Level 3 - Go further

11. **Semantic search.** `pip install -r requirements-optional.txt`, set `EMBEDDER=sentence-transformers`, and ask "Can I use ChatGPT for my essay?". Compare with TF-IDF. Why is it better?
12. **A real LLM.** Install [Ollama](https://ollama.com), run `ollama pull llama3.2`, set `LLM_PROVIDER=ollama`. Ask "If I submit 2 days late, what percent do I lose?" Compare with the extractive answer.
13. **Hybrid search.** Combine TF-IDF scores and sentence-transformer scores (e.g. average them). Does it beat both on `lessons/07_evaluate.py`?
14. **Re-ranking.** Retrieve 10 chunks, then keep only the 3 that share the most keywords with the question. Does the answer improve?
15. **Chat memory.** Make follow-up questions work ("What about Sundays?" after asking about library hours) by adding the previous question to the new search query.
16. **Use a real vector database.** Replace `VectorStore` with [ChromaDB](https://docs.trychroma.com) (`pip install chromadb`). Keep the same `add()` and `search()` methods so the rest of the code doesn't change.

## Discussion questions

- Why does RAG fail at "What is the average mark of the class?" What tool would you use instead?
- Try it: in `data/student_marks.csv` change one student's club to "Football Team", then ask "Who won the 2022 football world cup?". Why does adding an unrelated file change the answer to a different question?
- Why can't we just paste ALL documents into the prompt?
- What are the risks if the documents contain wrong or outdated information?
- How could someone "attack" a RAG system by adding a document with hidden instructions? (Search: *prompt injection*.)
- When is RAG better than fine-tuning a model, and when is fine-tuning better?
