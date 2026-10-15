# RAG Classroom: Learn Retrieval-Augmented Generation by Building It

A complete, beginner-friendly RAG project for teaching students. Every part of a RAG system is written in plain, commented Python so students can read it, change it and see what happens.

**This copy is set up to use OpenAI.** OpenAI embeddings find the right text, and an OpenAI chat model writes the answer. It's in **hybrid mode**: ask about your documents and you get an answer with sources; ask anything else and OpenAI answers from its general knowledge, labelled as such, like ChatGPT. **The only thing you change is your API key, in one line of the `.env` file** (see step 4 below).

No key? Two lines in `.env` switch it to a 100% free, offline mode (see [Running without a key](#running-without-a-key)).

```
 Question ─► Embed ─► Search vector store ─► Top-k chunks ─► Prompt ─► LLM ─► Answer + sources
                          ▲
 Documents ─► Load ─► Chunk ─► Embed ─► Store   (done once: "indexing")
```

---

## 1. Quick start (5 minutes)

**You need:** Python 3.10 or newer ([python.org](https://www.python.org/downloads/), on Windows tick *"Add python.exe to PATH"*) and [VS Code](https://code.visualstudio.com/) with the **Python** extension.

1. Unzip the folder and open it in VS Code: **File → Open Folder… → `rag-for-students`**
2. Open a terminal in VS Code: **Terminal → New Terminal**
3. Run the setup script:

   | Windows (PowerShell, the VS Code default) | Windows (Command Prompt) | macOS / Linux |
   |---|---|---|
   | `.\setup.bat` | `setup.bat` | `bash setup.sh` |

   PowerShell only runs scripts in the current folder when you type `.\` in front.

   Or do it by hand:

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

4. **Paste your OpenAI key. This is the only change you need to make.**

   - In VS Code's file list on the left, click the file **`.env`** (in the main folder, next to `app.py`).
     *Can't see it?* Files starting with a dot can be hidden. Use **File → Open File…** and type `.env`.
   - Find **line 9**:

     ```
     OPENAI_API_KEY=paste-your-openai-key-here
     ```

   - Replace `paste-your-openai-key-here` with your key, so it looks like:

     ```
     OPENAI_API_KEY=sk-proj-AbC123...your-full-key...
     ```

     No quotes, no spaces around `=`. Then save with **Ctrl+S**.

5. In VS Code press **Ctrl+Shift+P → "Python: Select Interpreter"** and choose the one in `.venv`.
6. Run it:

   ```bash
   python app.py                 # chat in the terminal
   streamlit run web_app.py      # chat in the browser (http://localhost:8501)
   python -m pytest -v           # run the tests
   ```

> **Windows PowerShell says "running scripts is disabled"** when you activate? Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and answer Y. Or skip activation and call the venv directly: `.venv\Scripts\python app.py`.

Quick check that your key works:

```bash
python app.py -q "If I submit my assignment 2 days late, how many marks do I lose?"
```

You should get a written answer such as *"You would lose 20% of the marks (10% per day) [1]"*.

### Keep your key safe
- The key lives **only** in `.env`. It is never in the code, and `.env` is excluded from Git by `.gitignore`.
- **Don't give students this folder with your key inside.** Before sharing, put `paste-your-openai-key-here` back on line 9.
- If a key leaks, delete it in your OpenAI account and create a new one.

### Running without a key
Change these two lines in `.env`:
```
LLM_PROVIDER=extractive
EMBEDDER=tfidf
```
The project then runs free and offline. Search uses word matching, and answers are the best-matching sentences copied from the documents. This is good for students without a key.

Try asking: *"What is the late submission policy?"*, *"When does the library close on Saturday?"*, *"Why is Venus so hot?"*, *"Who won the world cup?"* (not in the documents, so it should say it doesn't know).

---

## 2. Project structure

```
rag-for-students/
├── README.md                ← you are here
├── requirements.txt         ← packages to install
├── requirements-optional.txt← neural embeddings (sentence-transformers)
├── .env.example             ← settings template (copy to .env)
├── setup.bat / setup.sh     ← one-click setup
│
├── data/                    ← THE DOCUMENTS. Drop your own .txt/.md/.pdf here!
│   ├── greenfield_college_handbook.md   (fictional - an LLM can't know it)
│   ├── greenfield_courses.csv           (course timetable - a table)
│   ├── student_marks.csv                (fictional marks - shows RAG's limits)
│   ├── greenfield_exam_notice.docx      (a Word file)
│   ├── solar_system.md
│   ├── photosynthesis.md
│   └── python_basics.md
│
├── rag/                     ← THE LIBRARY: one file per RAG step
│   ├── config.py            settings (read from .env)
│   ├── loader.py            1. load files (.txt .md .pdf .csv .docx)
│   ├── chunker.py           2. split into chunks
│   ├── embedder.py          3. text → vectors (TF-IDF or sentence-transformers)
│   ├── vector_store.py      4. store vectors + similarity search
│   ├── generator.py         5. prompt + answer (extractive / Ollama / Claude / OpenAI)
│   └── pipeline.py          all steps glued together
│
├── lessons/                 ← STEP-BY-STEP TEACHING SCRIPTS (run in order)
│   ├── 01_why_rag.py
│   ├── 02_load_and_chunk.py
│   ├── 03_embeddings.py
│   ├── 04_vector_search.py
│   ├── 05_prompt_building.py
│   ├── 06_generation.py
│   ├── 07_evaluate.py
│   └── 08_tables_and_limits.py
│
├── app.py                   ← terminal chat app
├── web_app.py               ← browser chat app (Streamlit)
├── evaluation/questions.json← test questions for lesson 7
├── exercises/EXERCISES.md   ← 14 exercises + discussion questions
├── tests/test_rag.py        ← automated tests (pytest)
└── .vscode/                 ← run/debug buttons and test settings for VS Code
```

---

## 3. Suggested lesson plan

Each lesson is a short script that prints explanations as it runs and pauses so students can read. Run with `python lessons/01_why_rag.py`, or open the file in VS Code and press **F5** (the "Run current lesson / file" configuration is preselected).

| # | Lesson | Students learn | Time |
|---|---|---|---|
| 1 | `01_why_rag.py` | Hallucination; the retrieve → augment → generate idea | 15 min |
| 2 | `02_load_and_chunk.py` | Loading files, chunk size, overlap | 20 min |
| 3 | `03_embeddings.py` | Vectors, cosine similarity, bag-of-words by hand, TF-IDF, and the limits of word matching | 30 min |
| 4 | `04_vector_search.py` | Building an index, top-k search, similarity scores, saving the index | 20 min |
| 5 | `05_prompt_building.py` | System prompts, numbered context, citations, "I don't know" | 15 min |
| 6 | `06_generation.py` | Extractive answers vs. a real LLM | 20 min |
| 7 | `07_evaluate.py` | Measuring hit rate and answer quality, tuning settings | 25 min |
| 8 | `08_tables_and_limits.py` | CSV and Word files; why RAG can't compute averages or "highest" over a table | 20 min |

Then: `app.py` / `web_app.py` for free exploration, and `exercises/EXERCISES.md` for homework.

**Debugging tip for class:** put a breakpoint (click left of a line number) in `rag/vector_store.py` inside `search()`, start the "Chat app (terminal)" configuration from the **Run and Debug** panel, ask a question, and inspect `scores` live.

---

## 4. Supported file types

Put any of these in `data/`. Each is turned into plain text by `rag/loader.py`:

| Type | How it's read |
|---|---|
| `.md`, `.txt` | As is |
| `.pdf` | Text from each page (scanned/image PDFs have no text) |
| `.docx` (Word) | Paragraphs and tables; headings kept as headings. Old `.doc` files: save as `.docx` first |
| `.csv` | Each **row** becomes a sentence with the column names, e.g. `student: Asha Verma \| maths: 88 \| ...` |

**Tables:** RAG is great for look-ups ("What did Asha score in maths?") but **can't** answer whole-table questions ("average mark", "who scored highest"), because it only reads the top few rows it retrieves. Lesson 8 shows this.

## 5. Answer mode: documents only, or documents + general knowledge

Set `ANSWER_MODE` on line 29 of `.env`, choose it in the web app's sidebar, or run `python app.py --mode strict`:

| Mode | Question about the documents | Any other question |
|---|---|---|
| `hybrid` (**this copy's setting**) | Answered from the documents, with sources | Answered from the AI's general knowledge, starting with *"From general knowledge:"* |
| `strict` | Answered from the documents, with sources | *"I don't know based on the provided documents."* |

The only difference between the two is the instruction text sent to the model (see lesson 5). That makes it a good way to show students how much the prompt controls.

Hybrid mode needs a real LLM. With `LLM_PROVIDER=extractive`, there is no general knowledge to fall back on.

## 6. Choosing the language model

Edit `LLM_PROVIDER` in `.env`:

| Provider | Cost | Setup |
|---|---|---|
| `extractive` (default) | Free, offline | Nothing. Picks the best-matching sentences; can't reason or rephrase. |
| `ollama` | Free, runs on your laptop | Install [Ollama](https://ollama.com), then `ollama pull llama3.2`. Needs ~4 GB free RAM. |
| `anthropic` | Paid API | Put `ANTHROPIC_API_KEY=...` in `.env`. Model set with `ANTHROPIC_MODEL`. |
| `openai` (**this copy's setting**) | Paid API, very cheap with small models | Put `OPENAI_API_KEY=...` in `.env`. Model set with `OPENAI_MODEL`. Also works with any OpenAI-compatible service via `OPENAI_BASE_URL`. |

You can also switch per run: `python app.py --provider ollama`.

## 7. Choosing the embedder

| `EMBEDDER` | What it does |
|---|---|
| `openai` (**this copy's setting**) | OpenAI's `text-embedding-3-small`. Understands meaning. Each text's embedding is saved in `index/openai_embedding_cache.json`, so you only pay once per text. |
| `tfidf` | Word-based. Instant, offline, free. Fails on synonyms ("ChatGPT" vs "AI tools"). |
| `sentence-transformers` | Neural, understands meaning. `pip install -r requirements-optional.txt` (large download: PyTorch + ~90 MB model on first run). |

---

## 8. Using it as a library

```python
from rag import RAGPipeline, Config

rag = RAGPipeline(Config(top_k=4, chunk_size=400))
rag.ingest("data")                       # load → chunk → embed → store
answer = rag.ask("What is the passing mark?")

print(answer.text)
for r in answer.sources:
    print(r.chunk.source, round(r.score, 3))
print(answer.prompt)                     # exactly what the LLM saw
```

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'rag'` | Run commands from the project folder (the one containing `app.py`). |
| `ModuleNotFoundError: sklearn / numpy / streamlit` | Activate the venv (`.venv\Scripts\activate` or `source .venv/bin/activate`) and `pip install -r requirements.txt`. Check VS Code is using the `.venv` interpreter. |
| `setup.bat is not recognized` (PowerShell) | Type `.\setup.bat` instead. |
| `'python' is not recognized` (Windows) | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python`. |
| Ollama: "Could not reach http://localhost:11434" | Open the Ollama app (or run `ollama serve`) and `ollama pull llama3.2`. |
| A general question gets "I don't know" | `ANSWER_MODE` is `strict`. Set it to `hybrid` on line 29 of `.env`. |
| `OpenAI key not set` | You haven't replaced `paste-your-openai-key-here` on line 9 of `.env`, or you didn't save the file. |
| `API error 401` | The key is wrong. Copy it again from your OpenAI account, with no extra spaces. |
| `API error 429` | No credit or billing on your OpenAI account, or too many requests in a short time. |
| `API error 404` mentioning a model | Change `OPENAI_MODEL` in `.env` to a model your account can use ([model list](https://platform.openai.com/docs/models)). |
| `Could not reach api.openai.com` | No internet, or a college firewall or proxy is blocking it. Try a mobile hotspot, or use the free mode. |
| `No module named 'docx'` | `pip install -r requirements.txt` again (it now includes `python-docx`). |
| A PDF gives empty text | It's a scanned image. Use a PDF with selectable text, or OCR it first. |
| Answers are always "I don't know" | Lower `MIN_SCORE` in `.env`, or check that your files are in `data/`. |
