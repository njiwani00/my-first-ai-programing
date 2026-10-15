"""
Automated tests. Run:  python -m pytest -v
(or in VS Code: Testing panel on the left -> Run All)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import Config, RAGPipeline, VectorStore, chunk_text, load_documents  # noqa: E402
from rag.chunker import Chunk, split_sentences  # noqa: E402
from rag.loader import load_file  # noqa: E402
from rag.embedder import TfidfEmbedder  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def test_load_documents_finds_sample_files():
    docs = load_documents(DATA)
    assert len(docs) >= 4
    assert all(d.text for d in docs)


def test_all_file_types_are_loaded():
    sources = {d.source for d in load_documents(DATA)}
    assert any(s.endswith(".md") for s in sources)
    assert any(s.endswith(".csv") for s in sources)
    assert any(s.endswith(".docx") for s in sources)


def test_csv_rows_become_sentences_with_column_names():
    doc = load_file(DATA / "student_marks.csv")
    assert "student: Sneha Iyer | roll_no: GC2026-03 | maths: 95" in doc.text


def test_docx_headings_are_marked(tmp_path):
    doc = load_file(DATA / "greenfield_exam_notice.docx")
    assert "## Hall Tickets" in doc.text
    assert "750 rupees" in doc.text


def test_sentences_are_not_split_after_titles():
    parts = split_sentences("The teacher is Dr. Meera Rao. She teaches CS201.")
    assert parts == ["The teacher is Dr. Meera Rao.", "She teaches CS201."]


def test_chunks_respect_size_and_cover_text():
    text = "This is a sentence about RAG. " * 100
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("hello world.", chunk_size=100, overlap=100)


def test_embeddings_are_normalised_and_similar_texts_score_higher():
    emb = TfidfEmbedder()
    texts = ["the library closes at ten", "venus is the hottest planet"]
    emb.fit(texts)
    vecs = emb.embed(texts)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    q = emb.embed(["when does the library close"])[0]
    assert vecs[0] @ q > vecs[1] @ q


def test_vector_store_search_and_save(tmp_path):
    store = VectorStore()
    chunks = [Chunk("a", "x.md", 0), Chunk("b", "y.md", 0)]
    store.add(chunks, np.array([[1.0, 0.0], [0.0, 1.0]]))
    results = store.search(np.array([0.9, 0.1]), top_k=1)
    assert results[0].chunk.source == "x.md"

    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 2


@pytest.fixture(scope="module")
def rag():
    pipeline = RAGPipeline(Config(llm_provider="extractive", embedder="tfidf"))
    pipeline.ingest(str(DATA))
    return pipeline


@pytest.mark.parametrize(
    "question, source, expected",
    [
        ("What is the fine for a late library book?", "greenfield_college_handbook.md", "10 rupees"),
        ("Which planet is the hottest?", "solar_system.md", "Venus"),
        ("Where does the Calvin cycle happen?", "photosynthesis.md", "stroma"),
        ("How do I create a virtual environment?", "python_basics.md", "venv"),
        ("Who teaches Data Structures?", "greenfield_courses.csv", "Meera Rao"),
        ("What did Sneha Iyer score in programming?", "student_marks.csv", "programming: 97"),
        ("How much is the revaluation fee?", "greenfield_exam_notice.docx", "750 rupees"),
    ],
)
def test_end_to_end_answers(rag, question, source, expected):
    answer = rag.ask(question)
    assert source in [r.chunk.source for r in answer.sources]
    assert expected.lower() in answer.text.lower()


def test_unknown_question_says_i_dont_know(rag):
    answer = rag.ask("Who won the 2022 football world cup?")
    assert "don't know" in answer.text.lower()


# ---------------- OpenAI wiring (no real API calls: the HTTP layer is faked) ----------------
def test_placeholder_key_gives_clear_message(monkeypatch):
    from rag import api

    monkeypatch.setenv("OPENAI_API_KEY", "paste-your-openai-key-here")
    with pytest.raises(RuntimeError, match="OpenAI key not set"):
        api.openai_key()


def test_openai_pipeline_with_fake_api(monkeypatch, tmp_path):
    from rag import api
    from rag.embedder import OpenAIEmbedder

    calls = {"embeddings": 0, "chat": 0}

    def fake_post(url, payload, headers, timeout=120):
        assert headers["Authorization"] == "Bearer sk-test"
        if url.endswith("/embeddings"):
            calls["embeddings"] += 1
            # fake "meaning": a vector from letter counts, good enough to test the plumbing
            vecs = [[t.lower().count(c) + 0.01 for c in "abcdefghijklmnopqrstuvwxyz"] for t in payload["input"]]
            return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]}
        calls["chat"] += 1
        assert "Context passages" in payload["messages"][1]["content"]
        return {"choices": [{"message": {"content": "Fake answer [1]"}}]}

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(api, "post_json", fake_post)

    pipeline = RAGPipeline(Config(llm_provider="openai", embedder="openai"))
    assert isinstance(pipeline.embedder, OpenAIEmbedder)
    # use an empty cache in a temporary folder, so the test never touches index/
    pipeline.embedder.cache_file = tmp_path / "cache.json"
    pipeline.embedder.cache = {}
    pipeline.ingest(str(DATA))
    answer = pipeline.ask("What is the passing mark?")
    assert answer.text == "Fake answer [1]"
    assert calls["chat"] == 1 and calls["embeddings"] >= 2

    # Embeddings are cached: asking the same question again makes no new embedding call
    before = calls["embeddings"]
    pipeline.ask("What is the passing mark?")
    assert calls["embeddings"] == before


# ---------------- Answer modes ----------------
def test_answer_mode_changes_only_the_instructions():
    from rag.generator import HYBRID_SYSTEM_PROMPT, SYSTEM_PROMPT, system_prompt

    assert system_prompt(Config(answer_mode="strict")) == SYSTEM_PROMPT
    assert system_prompt(Config(answer_mode="hybrid")) == HYBRID_SYSTEM_PROMPT
    assert "general knowledge" in HYBRID_SYSTEM_PROMPT.lower()


@pytest.mark.parametrize(
    "model_reply, expected_used_documents",
    [
        ("The passing mark is 40 percent overall. [1]", True),
        ("From general knowledge: The capital of Australia is Canberra.", False),
    ],
)
def test_hybrid_answers_are_labelled(monkeypatch, model_reply, expected_used_documents):
    from rag import api

    sent = {}

    def fake_post(url, payload, headers, timeout=120):
        sent["system"] = payload["messages"][0]["content"]
        return {"choices": [{"message": {"content": model_reply}}]}

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(api, "post_json", fake_post)

    pipeline = RAGPipeline(Config(llm_provider="openai", embedder="tfidf", answer_mode="hybrid"))
    pipeline.ingest(str(DATA))
    answer = pipeline.ask("What is the passing mark?")
    assert "general knowledge" in sent["system"].lower()
    assert answer.used_documents is expected_used_documents
