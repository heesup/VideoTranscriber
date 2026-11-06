"""
OpenAI-compatible local LLM integration for text summarization.

Works with any server exposing the OpenAI chat API (llama.cpp server,
LM Studio, vLLM, Strata, etc.). Configuration is read from environment
variables (also loaded from a .env file in the project root):

* OPENAI_COMPAT_BASE_URL - server base URL, default http://127.0.0.1:8080/v1
* OPENAI_COMPAT_API_KEY  - bearer token, optional for most local servers

Provides functions for text summarization with streaming support, mirroring
utils.ollama_integration.
"""

import requests
import json
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

DEFAULT_BASE_URL = os.environ.get("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:8080/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_COMPAT_API_KEY", "")

# Reasoning models spend part of the token budget on thinking before
# answering, so keep a generous ceiling on top of the requested length.
MIN_MAX_TOKENS = 2048


def _base_url(base_url=None):
    return (base_url or DEFAULT_BASE_URL).rstrip("/")


def _headers(api_key=None):
    headers = {"Content-Type": "application/json"}
    key = api_key if api_key is not None else DEFAULT_API_KEY
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def check_openai_compatible_available(base_url=None, api_key=None):
    """Check if an OpenAI-compatible service is reachable."""
    try:
        response = requests.get(
            f"{_base_url(base_url)}/models",
            headers=_headers(api_key),
            timeout=2,
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models(base_url=None, api_key=None):
    """List model IDs exposed by the OpenAI-compatible service."""
    try:
        response = requests.get(
            f"{_base_url(base_url)}/models",
            headers=_headers(api_key),
            timeout=5,
        )
        if response.status_code == 200:
            return [model["id"] for model in response.json().get("data", [])]
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error listing OpenAI-compatible models: {e}")
        return []


def _extract_content(message):
    """Return the answer text; fall back to reasoning text for thinking models."""
    content = (message.get("content") or "").strip()
    if content:
        return content
    reasoning = (message.get("reasoning_content") or "").strip()
    if reasoning:
        logger.warning("Model returned no content; using reasoning output")
        return reasoning
    return ""


def summarize_with_openai_compatible(
    text, model, max_length=150, base_url=None, api_key=None
):
    """Summarize text via /chat/completions (non-streaming)."""
    if not check_openai_compatible_available(base_url, api_key):
        logger.warning("OpenAI-compatible service is not available")
        return None

    prompt = f"Summarize the following text in about {max_length} words:\n\n{text}"

    try:
        response = requests.post(
            f"{_base_url(base_url)}/chat/completions",
            headers=_headers(api_key),
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": max(max_length * 2, MIN_MAX_TOKENS),
            },
            timeout=300,
        )

        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            if choices:
                return _extract_content(choices[0].get("message", {}))
            return None
        else:
            logger.error(
                f"OpenAI-compatible API error: {response.status_code} - {response.text}"
            )
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with OpenAI-compatible service: {e}")
        return None


def stream_summarize_with_openai_compatible(
    text, model, max_length=150, base_url=None, api_key=None
):
    """
    Summarize text with streaming. Yields tokens as they arrive.

    Yields:
        str: Individual response tokens
    """
    if not check_openai_compatible_available(base_url, api_key):
        logger.warning("OpenAI-compatible service is not available")
        return

    prompt = f"Summarize the following text in about {max_length} words:\n\n{text}"

    try:
        response = requests.post(
            f"{_base_url(base_url)}/chat/completions",
            headers=_headers(api_key),
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": max(max_length * 2, MIN_MAX_TOKENS),
            },
            stream=True,
            timeout=300,
        )

        if response.status_code == 200:
            reasoning_buffer = []
            yielded_any = False
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content") or ""
                if token:
                    yielded_any = True
                    yield token
                else:
                    reasoning = delta.get("reasoning_content") or ""
                    if reasoning:
                        reasoning_buffer.append(reasoning)
            if not yielded_any and reasoning_buffer:
                logger.warning("Stream returned no content; using reasoning output")
                yield from reasoning_buffer
        else:
            logger.error(f"OpenAI-compatible API error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with OpenAI-compatible service: {e}")


def chunk_and_summarize(
    text, model, chunk_size=4000, max_length=150, base_url=None, api_key=None
):
    """Chunk long text and summarize each chunk, then combine."""
    if len(text) <= chunk_size:
        return summarize_with_openai_compatible(text, model, max_length, base_url, api_key)

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_with_openai_compatible(
            chunk, model, max_length // len(chunks), base_url, api_key
        )
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return None

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined_summary = " ".join(chunk_summaries)
    return summarize_with_openai_compatible(
        combined_summary, model, max_length, base_url, api_key
    )


def stream_chunk_and_summarize(
    text, model, chunk_size=4000, max_length=150, base_url=None, api_key=None
):
    """
    Chunk and summarize with streaming on the final summary.
    Returns non-streaming chunk summaries, then streams the final combination.

    Yields:
        str: Tokens from the final summary
    """
    if len(text) <= chunk_size:
        yield from stream_summarize_with_openai_compatible(
            text, model, max_length, base_url, api_key
        )
        return

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_with_openai_compatible(
            chunk, model, max_length // len(chunks), base_url, api_key
        )
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return

    if len(chunk_summaries) == 1:
        yield chunk_summaries[0]
        return

    combined_summary = " ".join(chunk_summaries)
    yield from stream_summarize_with_openai_compatible(
        combined_summary, model, max_length, base_url, api_key
    )
