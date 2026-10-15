"""
Small helpers for calling AI APIs over HTTP.

Only Python's built-in urllib is used, so no extra packages are needed.
"""

import json
import os
import urllib.error
import urllib.request

PLACEHOLDER_KEYS = {"", "paste-your-openai-key-here", "your-key-here", "sk-...", "sk-xxxx"}


class APIError(RuntimeError):
    def __init__(self, message: str, status: int = 0, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="ignore")
        raise APIError(f"API error {err.code}: {_explain(err.code)}\n{body[:400]}", err.code, body) from err
    except urllib.error.URLError as err:
        raise APIError(f"Could not reach {url}: {err.reason}. Check your internet connection.") from err


def _explain(status: int) -> str:
    return {
        401: "the API key is wrong. Check OPENAI_API_KEY in your .env file.",
        403: "this key is not allowed to use this model or endpoint.",
        404: "model not found. Check the model name in your .env file.",
        429: "rate limit or no credit left. Check billing on your OpenAI account.",
    }.get(status, "see the message below.")


def openai_key() -> str:
    """Read the OpenAI key from .env and give a clear message if it's missing."""
    key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
    if key.lower() in PLACEHOLDER_KEYS:
        raise RuntimeError(
            "OpenAI key not set.\n"
            "  1. Open the file .env in the project folder\n"
            "  2. Replace paste-your-openai-key-here with your real key (starts with sk-)\n"
            "  3. Save the file and run again.\n"
            "No key? Set LLM_PROVIDER=extractive and EMBEDDER=tfidf in .env to run for free."
        )
    return key
