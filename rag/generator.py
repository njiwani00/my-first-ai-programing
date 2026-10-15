"""
STEP 5 - GENERATE: answer the question using the retrieved chunks.

Two answer modes (ANSWER_MODE in .env):
  strict - answer ONLY from the documents, otherwise "I don't know". Classic RAG.
  hybrid - answer from the documents when they contain the answer (with sources);
           otherwise answer from the model's general knowledge, clearly labelled
           "From general knowledge:". Works like a normal chatbot PLUS your documents.

The "augmented" in Retrieval-AUGMENTED Generation is the prompt below:
we paste the relevant chunks into the prompt, number them, and instruct the
model to answer from them and cite [1], [2], ...

Providers (choose with LLM_PROVIDER in .env):
  extractive  - NO language model, NO API key. Picks the best-matching
                sentences from the chunks. Great for class: free and offline.
  anthropic   - Claude models  (needs ANTHROPIC_API_KEY)
  openai      - OpenAI or any OpenAI-compatible API, e.g. Groq, Together
                (needs OPENAI_API_KEY, optional OPENAI_BASE_URL)
  ollama      - free local models on your own computer (https://ollama.com)

All HTTP calls use Python's built-in urllib, so no extra packages are needed.
"""

import os
import re
from typing import List

from . import api
from .chunker import SENTENCE_END
from .config import Config
from .vector_store import SearchResult

# ANSWER_MODE=strict : answer ONLY from the documents (classic RAG, no hallucination)
SYSTEM_PROMPT = (
    "You are a helpful teaching assistant. Answer the question using ONLY the "
    "numbered context passages. Cite the passages you used like [1] or [2]. "
    "If the answer is not in the context, say \"I don't know based on the provided documents.\" "
    "Keep the answer short and clear."
)

# ANSWER_MODE=hybrid : documents first, general knowledge when they don't help
GENERAL_KNOWLEDGE_LABEL = "From general knowledge:"
HYBRID_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. You are given numbered context passages "
    "from the user's documents, followed by a question.\n"
    "1. If the passages contain the answer, answer from them and cite them like [1] or [2]. "
    "The documents are the authority: if they disagree with what you know, follow the documents.\n"
    f"2. If the passages are not relevant or don't contain the answer, IGNORE them and answer "
    f"from your own general knowledge. Start that answer with exactly \"{GENERAL_KNOWLEDGE_LABEL}\" "
    "and do not cite any passage.\n"
    "3. If the question is a greeting or small talk, just reply naturally.\n"
    "Be clear and helpful."
)


def system_prompt(config: Config) -> str:
    return HYBRID_SYSTEM_PROMPT if config.answer_mode == "hybrid" else SYSTEM_PROMPT


def is_general_knowledge(answer_text: str) -> bool:
    """True if the model answered from its own knowledge instead of the documents."""
    return answer_text.strip().lower().startswith(GENERAL_KNOWLEDGE_LABEL.lower())


def build_prompt(question: str, results: List[SearchResult]) -> str:
    """Create the augmented prompt: context passages + the question."""
    if not results:
        context = "(no relevant passages were found)"
    else:
        context = "\n\n".join(
            f"[{i}] (source: {r.chunk.source})\n{r.chunk.text}" for i, r in enumerate(results, start=1)
        )
    return f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"


# --------------------------------------------------------------------------
# 1) Extractive "generator" - no LLM at all
# --------------------------------------------------------------------------
_STOPWORDS = set(
    """a an the is are was were be been being am do does did of to in on at for from by with
    and or but if then so than that this these those it its as what which who whom whose
    when where why how can could should would will shall may might must i you he she we they
    me my your our their there here about into over under up down out not no yes any all
    some much many more most tell explain give please""".split()
)


def _keywords(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _shared(question_words: set, sentence_words: set) -> int:
    """Count question words found in the sentence. 'hot' also matches 'hottest'."""
    count = 0
    for q in question_words:
        if any(q == s or (len(q) >= 3 and len(s) >= 3 and (s.startswith(q) or q.startswith(s)))
               for s in sentence_words):
            count += 1
    return count


class ExtractiveGenerator:
    """Picks the sentences that share the most keywords with the question."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, results: List[SearchResult]) -> str:
        if not results:
            return "I don't know based on the provided documents."

        q_words = _keywords(question)
        candidates = []
        for idx, r in enumerate(results, start=1):
            for sentence in re.split(SENTENCE_END + r"|\n", r.chunk.text):
                sentence = sentence.strip()
                if len(sentence) < 20 or sentence.startswith("#"):  # skip headings
                    continue
                overlap = _shared(q_words, _keywords(sentence))
                if overlap == 0:
                    continue
                # more shared keywords + higher chunk similarity = better
                candidates.append((overlap + r.score, overlap, idx, sentence))

        if not candidates:
            return "I don't know based on the provided documents."

        candidates.sort(key=lambda c: -c[0])
        best_overlap = candidates[0][1]
        picked, seen = [], set()
        for _, overlap, idx, sentence in candidates:
            if overlap < best_overlap / 2:  # much weaker match than the best: skip
                continue
            if sentence not in seen:
                picked.append(f"{sentence} [{idx}]")
                seen.add(sentence)
            if len(picked) == self.max_sentences:
                break
        return " ".join(picked)


# --------------------------------------------------------------------------
# 2) Real language models over HTTP
# --------------------------------------------------------------------------
class AnthropicGenerator:
    name = "anthropic"

    def __init__(self, config: Config):
        self.config = config
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY in your .env file (or use LLM_PROVIDER=extractive).")

    def generate(self, question: str, results: List[SearchResult]) -> str:
        payload = {
            "model": self.config.anthropic_model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system_prompt(self.config),
            "messages": [{"role": "user", "content": build_prompt(question, results)}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        data = api.post_json("https://api.anthropic.com/v1/messages", payload, headers)
        return "".join(block.get("text", "") for block in data.get("content", [])).strip()


class OpenAIGenerator:
    name = "openai"

    def __init__(self, config: Config):
        self.config = config
        self.api_key = api.openai_key()

    def generate(self, question: str, results: List[SearchResult]) -> str:
        payload = {
            "model": self.config.openai_model,
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt(self.config)},
                {"role": "user", "content": build_prompt(question, results)},
            ],
        }
        url = self.config.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            data = api.post_json(url, payload, headers)
        except api.APIError as err:
            # Some newer models only accept their default temperature: retry without it.
            if err.status == 400 and "temperature" in err.body:
                payload.pop("temperature")
                data = api.post_json(url, payload, headers)
            else:
                raise
        return (data["choices"][0]["message"]["content"] or "").strip()


class OllamaGenerator:
    name = "ollama"

    def __init__(self, config: Config):
        self.config = config

    def generate(self, question: str, results: List[SearchResult]) -> str:
        payload = {
            "model": self.config.ollama_model,
            "stream": False,
            "options": {"temperature": self.config.temperature},
            "messages": [
                {"role": "system", "content": system_prompt(self.config)},
                {"role": "user", "content": build_prompt(question, results)},
            ],
        }
        url = self.config.ollama_url.rstrip("/") + "/api/chat"
        try:
            data = api.post_json(url, payload, {})
        except RuntimeError as err:
            raise RuntimeError(
                f"{err}\nIs Ollama running? Install from https://ollama.com then run: "
                f"ollama pull {self.config.ollama_model}"
            ) from err
        return data["message"]["content"].strip()


def get_generator(config: Config):
    provider = config.llm_provider.lower().strip()
    if provider == "extractive":
        return ExtractiveGenerator()
    if provider == "anthropic":
        return AnthropicGenerator(config)
    if provider == "openai":
        return OpenAIGenerator(config)
    if provider == "ollama":
        return OllamaGenerator(config)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use extractive, anthropic, openai or ollama.")
