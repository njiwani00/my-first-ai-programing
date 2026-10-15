"""
A web chat UI for the RAG system, built with Streamlit.

Install once:   pip install streamlit
Run:            streamlit run web_app.py
Then open the link it prints (usually http://localhost:8501).

Features:
  * chat with your documents, with sources shown under every answer
  * upload your own .txt / .md / .pdf files from the sidebar
  * change chunk size, top_k and the LLM provider live
  * "Show prompt" to see exactly what the model received
"""

import shutil
import tempfile
from pathlib import Path

import streamlit as st

from rag import Config, RAGPipeline
from rag.config import PROJECT_ROOT

st.set_page_config(page_title="RAG Classroom", page_icon="📚", layout="wide")
st.title("📚 RAG Classroom - chat with your documents")
st.caption("Hybrid mode: ask about the documents or anything else. Strict mode: documents only.")

# ---------------- Sidebar: settings ----------------
defaults = Config()
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox(
        "LLM provider",
        ["extractive", "ollama", "anthropic", "openai"],
        index=["extractive", "ollama", "anthropic", "openai"].index(defaults.llm_provider)
        if defaults.llm_provider in ("extractive", "ollama", "anthropic", "openai")
        else 0,
        help="extractive needs no API key. Keys go in the .env file.",
    )
    mode_labels = {"hybrid": "Documents + general knowledge", "strict": "Documents only (strict RAG)"}
    answer_mode = st.radio(
        "Answer mode",
        ["hybrid", "strict"],
        index=0 if defaults.answer_mode == "hybrid" else 1,
        format_func=mode_labels.get,
        help="Hybrid: answers from your documents when they contain the answer, otherwise from the "
        "AI's general knowledge. Strict: only from the documents, otherwise 'I don't know'.",
    )
    chunk_size = st.slider("Chunk size (characters)", 100, 2000, defaults.chunk_size, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, chunk_size // 2, min(defaults.chunk_overlap, chunk_size // 2), step=10)
    top_k = st.slider("Top k chunks", 1, 10, defaults.top_k)
    show_prompt = st.checkbox("Show the prompt sent to the model")

    st.header("Documents")
    uploads = st.file_uploader("Add your own files", type=["txt", "md", "pdf", "csv", "docx"], accept_multiple_files=True)
    use_samples = st.checkbox("Include sample documents", value=True)


# ---------------- Build the index (cached) ----------------
@st.cache_resource(show_spinner="Building the index...")
def build(provider, chunk_size, chunk_overlap, top_k, upload_key, use_samples, _files):
    folder = Path(tempfile.mkdtemp(prefix="rag_docs_"))
    if use_samples:
        for f in (PROJECT_ROOT / "data").glob("*"):
            if f.is_file():
                shutil.copy(f, folder / f.name)
    for f in _files or []:
        (folder / f.name).write_bytes(f.getvalue())

    config = Config(chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k, llm_provider=provider)
    config.data_dir = str(folder)
    rag = RAGPipeline(config)
    count = rag.ingest()
    return rag, count


upload_key = tuple(sorted((f.name, f.size) for f in uploads)) if uploads else ()
try:
    rag, chunk_count = build(provider, chunk_size, chunk_overlap, top_k, upload_key, use_samples, uploads)
except Exception as err:  # noqa: BLE001 - show any problem to the student
    st.error(f"Could not build the index: {err}")
    st.stop()

# The answer mode only changes the instructions given to the model, not the index,
# so we can switch it without rebuilding.
rag.config.answer_mode = answer_mode

sources = sorted({c.source for c in rag.store.chunks})
st.caption(f"Indexed {len(sources)} files into {chunk_count} chunks: " + ", ".join(sources))

# ---------------- Chat ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("note"):
            st.caption(msg["note"])

examples = ["What is the late submission policy?", "Why is Venus so hot?", "What does chlorophyll do?"]
if answer_mode == "hybrid":
    examples.append("Explain photosynthesis to a 10 year old")
    examples.append("Write a Python function to reverse a string")
st.write("Try: " + " · ".join(f"*{e}*" for e in examples))

placeholder = "Ask anything" if answer_mode == "hybrid" else "Ask a question about the documents"
if question := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            answer = rag.ask(question)
        except RuntimeError as err:
            st.error(str(err))
            st.stop()
        st.markdown(answer.text)
        if answer.used_documents:
            note = "📄 Answered from your documents: " + ", ".join(dict.fromkeys(r.chunk.source for r in answer.sources))
            st.caption(note)
            with st.expander(f"Sources ({len(answer.sources)})"):
                for i, r in enumerate(answer.sources, 1):
                    st.markdown(f"**[{i}] {r.chunk.source}** - chunk {r.chunk.chunk_id}, similarity `{r.score:.3f}`")
                    st.text(r.chunk.text)
        else:
            note = "🌐 Answered from the AI's general knowledge (no document was used)"
            st.caption(note)
        if show_prompt:
            with st.expander("Prompt sent to the model"):
                st.code(answer.prompt, language="text")
    st.session_state.messages.append({"role": "assistant", "content": answer.text, "note": note})
