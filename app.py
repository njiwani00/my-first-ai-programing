"""
Chat with your documents in the terminal.

Run:
    python app.py                       # uses settings from .env
    python app.py --show-prompt         # also print the augmented prompt
    python app.py --mode strict         # documents only (classic RAG)
    python app.py --mode hybrid         # documents + general knowledge
    python app.py --provider ollama     # override the LLM provider
    python app.py --data my_notes       # use a different documents folder
    python app.py -q "What is the passing mark?"   # ask one question and exit

While chatting, type:
    :sources   show the full text of the chunks used for the last answer
    :prompt    show the prompt sent to the model for the last answer
    :quit      exit
"""

import argparse
import sys

from rag import Config, RAGPipeline


def print_answer(answer) -> None:
    print(f"\nAssistant: {answer.text}\n")
    if answer.used_documents:
        print("Sources:")
        for i, r in enumerate(answer.sources, 1):
            print(f"  [{i}] {r.chunk.source} (chunk {r.chunk.chunk_id}, score {r.score:.2f})")
    else:
        print("Source: general knowledge of the AI model (no document was used)")
    print(f"  ({answer.timings['retrieve']}s retrieve, {answer.timings['generate']}s generate)")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG chat over your documents")
    parser.add_argument("--data", help="folder with .txt/.md/.pdf files (default: data/)")
    parser.add_argument("--provider", help="extractive | anthropic | openai | ollama")
    parser.add_argument("--embedder", help="tfidf | sentence-transformers")
    parser.add_argument("--top-k", type=int, help="number of chunks to retrieve")
    parser.add_argument("--mode", choices=["strict", "hybrid"],
                        help="strict = documents only, hybrid = documents + general knowledge")
    parser.add_argument("--show-prompt", action="store_true", help="print the augmented prompt")
    parser.add_argument("-q", "--question", help="ask a single question and exit")
    args = parser.parse_args()

    config = Config()
    if args.data:
        config.data_dir = args.data
    if args.provider:
        config.llm_provider = args.provider
    if args.embedder:
        config.embedder = args.embedder
    if args.top_k:
        config.top_k = args.top_k
    if args.mode:
        config.answer_mode = args.mode

    print("Building the index...")
    try:
        rag = RAGPipeline(config)
        rag.ingest(verbose=True)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as err:
        print(f"Error: {err}")
        return 1
    print("Settings:", config.summary())

    def handle(question: str):
        try:
            answer = rag.ask(question)
        except RuntimeError as err:
            print(f"\nError: {err}")
            return None
        if args.show_prompt:
            print("\n----- PROMPT -----\n" + answer.prompt + "\n------------------")
        print_answer(answer)
        return answer

    if args.question:
        return 0 if handle(args.question) else 1

    if config.answer_mode == "hybrid" and config.llm_provider != "extractive":
        print("\nAsk anything: questions about your documents are answered from them (with sources),")
        print("other questions from the AI's general knowledge. Type :quit to exit.\n")
    else:
        print("\nAsk a question about your documents. Type :quit to exit.\n")
    if config.answer_mode == "hybrid" and config.llm_provider == "extractive":
        print("Note: hybrid mode needs a real LLM (e.g. LLM_PROVIDER=openai). The extractive answerer")
        print("has no general knowledge, so off-topic questions will get 'I don't know'.\n")
    last = None
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in (":quit", ":q", "exit", "quit"):
            break
        if question == ":sources":
            if last:
                for i, r in enumerate(last.sources, 1):
                    print(f"\n[{i}] {r.chunk.source}\n{r.chunk.text}")
            continue
        if question == ":prompt":
            if last:
                print(last.prompt)
            continue
        last = handle(question) or last
    print("Bye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
