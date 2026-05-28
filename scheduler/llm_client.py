"""
Thin OpenAI wrapper for the SuSchedule-r planner agent.

Features
--------
* Reads OPENAI_API_KEY from a .env file in the repo root (via python-dotenv),
  falling back to the environment variable if already set.
* ``call_structured(model, messages, schema)``
      Uses OpenAI Structured Outputs (beta.chat.completions.parse) to return
      a typed Pydantic object — the API guarantees the shape.
* ``call_chat(model, messages)``
      Plain text completion for cases where we don't need strict JSON
      (e.g. free-form explanations).
* Exponential back-off retry on rate-limit (429) and transient (5xx) errors.
* Token usage logged after every call for cost visibility.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError
from pydantic import BaseModel

# Load .env from repo root (harmless if already set in environment)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

T = TypeVar("T", bound=BaseModel)

# --------------------------------------------------------------------------- #
# Client singleton
# --------------------------------------------------------------------------- #

def _make_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it to .env in the repo root or export it in your shell."
        )
    return OpenAI(api_key=key)


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #

_RETRY_DELAYS = [1, 3, 8]   # seconds between attempts (3 attempts total)


def _with_retry(fn, *args, **kwargs):
    """Call *fn* with retries on rate-limit / transient server errors."""
    last_exc = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except RateLimitError as exc:
            print(f"[LLMClient] Rate limit hit (attempt {attempt}). Retrying in {delay}s…")
            last_exc = exc
        except APIStatusError as exc:
            if exc.status_code >= 500:
                print(f"[LLMClient] Server error {exc.status_code} (attempt {attempt}). Retrying…")
                last_exc = exc
            else:
                raise   # 4xx except 429 — don't retry
    raise last_exc  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Token usage logger
# --------------------------------------------------------------------------- #

_total_prompt_tokens = 0
_total_completion_tokens = 0


def _log_usage(usage, label: str) -> None:
    global _total_prompt_tokens, _total_completion_tokens
    if usage is None:
        return
    p, c = usage.prompt_tokens, usage.completion_tokens
    _total_prompt_tokens += p
    _total_completion_tokens += c
    print(
        f"[LLMClient] {label} | "
        f"prompt={p} completion={c} total={p+c} | "
        f"session_total={_total_prompt_tokens + _total_completion_tokens}"
    )


def get_session_usage() -> dict:
    return {
        "prompt_tokens": _total_prompt_tokens,
        "completion_tokens": _total_completion_tokens,
        "total_tokens": _total_prompt_tokens + _total_completion_tokens,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def call_structured(
    model: str,
    messages: list[dict],
    schema: Type[T],
    label: str = "structured",
) -> T:
    """Call the model and parse the response as a Pydantic *schema* object.

    Uses OpenAI's Structured Outputs (``beta.chat.completions.parse``).
    The API guarantees the returned JSON matches the schema exactly.

    Parameters
    ----------
    model:    e.g. ``"gpt-4o"`` or ``"gpt-4o-mini"``
    messages: standard OpenAI message list
    schema:   a Pydantic ``BaseModel`` subclass
    label:    short string for the usage log line

    Returns
    -------
    An instance of *schema* with all fields populated.
    """
    client = get_client()

    def _call():
        return client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
        )

    response = _with_retry(_call)
    _log_usage(response.usage, label)

    parsed = response.choices[0].message.parsed
    if parsed is None:
        # Should never happen with strict structured outputs, but be defensive.
        raw = response.choices[0].message.content or ""
        raise ValueError(
            f"[LLMClient] Structured output parse failed for {label}. "
            f"Raw content: {raw[:200]}"
        )
    return parsed


def call_chat(
    model: str,
    messages: list[dict],
    label: str = "chat",
) -> str:
    """Plain chat completion — returns the assistant's text content."""
    client = get_client()

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
        )

    response = _with_retry(_call)
    _log_usage(response.usage, label)
    return response.choices[0].message.content or ""


def call_with_tools(
    model: str,
    messages: list[dict],
    tools: list[dict],
    label: str = "tools",
    tool_choice: str = "auto",
):
    """Chat completion with function/tool calling.

    Returns the raw assistant ``message`` object so the caller can inspect
    ``message.tool_calls`` (list of ChatCompletionMessageToolCall) and
    ``message.content`` (final text, present when the model is done calling
    tools).
    """
    client = get_client()

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

    response = _with_retry(_call)
    _log_usage(response.usage, label)
    return response.choices[0].message
