"""
LESSON 8 - TABLES (CSV), WORD FILES, AND WHAT RAG CAN'T DO

Run:  python lessons/08_tables_and_limits.py

RAG works on TEXT. So every file type is first converted to plain text
(see rag/loader.py):

  .md / .txt   -> used as they are
  .pdf         -> text extracted from each page
  .docx        -> paragraphs joined; headings marked with "#"
  .csv         -> each ROW becomes one sentence that includes the column names

Why include the column names? A row like  "Sneha Iyer,GC2026-03,95,89,97"
means nothing on its own - which number is maths? So we write:
  "student: Sneha Iyer | roll_no: GC2026-03 | maths: 95 | physics: 89 | programming: 97"

Then we discover RAG's biggest limitation: it is a SEARCH tool, not a calculator.
"""

import csv

from _common import DATA_DIR, pause, short, title

from rag import RAGPipeline
from rag.loader import load_file

title("LESSON 8: Tables, Word files, and the limits of RAG")

# ---- 1. What a CSV looks like before and after loading ---------------------
csv_path = DATA_DIR / "student_marks.csv"
print("Raw CSV (first 3 lines):")
with open(csv_path, encoding="utf-8") as f:
    for _ in range(3):
        print("   ", f.readline().rstrip())

print("\nAfter loading, each row is a sentence:")
for line in load_file(csv_path).text.split("\n\n")[:2]:
    print("   ", line)
pause()

# ---- 2. A Word file ----------------------------------------------------------
title("A Word (.docx) file after loading")
print(load_file(DATA_DIR / "greenfield_exam_notice.docx").text[:500] + " ...")
pause()

# ---- 3. Questions RAG answers well: LOOK-UPS -------------------------------
rag = RAGPipeline()
rag.ingest()

title("Look-up questions (RAG is great at these)")
for q in [
    "What did Sneha Iyer score in programming?",
    "Who teaches Data Structures?",
    "How much is the revaluation fee?",
]:
    answer = rag.ask(q)
    print(f"\nQ: {q}\nA: {short(answer.text, 200)}")
pause()

# ---- 4. Questions RAG gets WRONG: calculations over the whole table --------
title("Whole-table questions (RAG fails at these)")
with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
true_average = sum(int(r["maths"]) for r in rows) / len(rows)
true_best = max(rows, key=lambda r: int(r["physics"]))

for q, truth in [
    ("What is the average maths mark of the class?", f"{true_average:.2f}"),
    ("Who scored the highest in physics?", f"{true_best['student']} ({true_best['physics']})"),
]:
    answer = rag.ask(q)
    print(f"\nQ: {q}")
    print(f"RAG answer : {short(answer.text, 160)}")
    print(f"Correct    : {truth}")
    print(f"Chunks seen: {len(answer.sources)} (out of {len(rows)} students in the table)")

title("Why did it fail?")
print(
    "* RAG only retrieves the top few chunks (top_k). To find an average or a maximum\n"
    "  you must look at EVERY row - RAG never sees most of them.\n"
    "* Even a perfect LLM can't compute an average from 3 of the 12 students.\n"
    "* Fun fact: for 'average', word search may even return the Solar System file,\n"
    "  because it contains the word 'average' (average distance to the Sun)!\n\n"
    "The right tool for whole-table questions is code, Excel or a database (SQL).\n"
    "Advanced systems let the LLM WRITE that code/SQL - search 'text-to-SQL'."
)
