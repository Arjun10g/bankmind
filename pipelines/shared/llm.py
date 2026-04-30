"""Claude LLM wrapper.

Single import surface for everything that calls Anthropic — query transformers,
RankGPT reranker, generation, QA pair generation, Track B reference answers.

Two layers:
  - `claude_text(prompt, ...)`        — most common case, returns a single string
  - `claude_json(prompt, schema=...)` — parse the response as JSON (with retry on
                                         malformed responses)

Default model: claude-sonnet-4-6 (good quality/latency tradeoff). Override
with `CLAUDE_MODEL` env var or per-call.

Caching: identical prompts return cached responses (in-memory LRU). Useful
during eval where the same query transform may be asked repeatedly. Disable
with `cache=False` per call.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
# Light, fast model for utility tasks like follow-up rewriting
FAST_MODEL = os.environ.get("CLAUDE_FAST_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_MAX_TOKENS = 1024


_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment / .env")
        _client = Anthropic(api_key=api_key)
    return _client


@lru_cache(maxsize=512)
def _cached_text(prompt: str, model: str, system: str, max_tokens: int, temperature: float) -> str:
    client = _get_client()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()


def claude_text(
    prompt: str,
    *,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    cache: bool = True,
) -> str:
    """Synchronous Claude call → single string response."""
    if cache:
        return _cached_text(prompt, model, system, max_tokens, temperature)
    # Cache-bypass path: re-implement the body since lru_cache can't be opted out per call
    client = _get_client()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()


def claude_text_stream(
    prompt: str,
    *,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
):
    """Generator yielding the response text as it arrives.

    Each yield is the *cumulative* output so far (suitable for piping straight
    into Gradio's Chatbot streaming). On error, yields a single error message.
    """
    try:
        client = _get_client()
        accumulated = ""
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for delta in stream.text_stream:
                accumulated += delta
                yield accumulated
    except Exception as e:
        yield f"_(generation failed: {type(e).__name__}: {e})_"


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def _strip_json_fences(text: str) -> str:
    m = _JSON_FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


def claude_json(
    prompt: str,
    *,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    cache: bool = True,
    retries: int = 2,
):
    """Claude call → parsed JSON. Retries on malformed responses with a stricter
    reminder."""
    last_err: Optional[Exception] = None
    sys_prompt = (system + "\n\n" if system else "") + (
        "Return ONLY valid JSON. No prose, no code fences, no commentary."
    )
    for attempt in range(retries + 1):
        text = claude_text(
            prompt,
            system=sys_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            cache=cache and attempt == 0,  # don't cache retries
        )
        try:
            return json.loads(_strip_json_fences(text))
        except json.JSONDecodeError as e:
            last_err = e
            # Retry path: re-prompt with the bad response and ask for clean JSON
            prompt = (
                f"Your previous response was not valid JSON:\n\n---\n{text}\n---\n\n"
                f"Return ONLY the JSON object/array. Do not wrap in code fences. "
                f"Original request:\n\n{prompt}"
            )
    raise ValueError(f"Claude failed to return valid JSON after {retries + 1} attempts: {last_err}")
