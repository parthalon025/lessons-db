"""Ollama and OpenAI API integration for the eval pipeline."""

import json as _json
import logging
import re as _re
import time
import urllib.error
import urllib.request
from typing import Any

from lessons_db.eval.variants import (
    _MAX_RETRIES,
    _RETRY_BASE_DELAY,
    _RETRYABLE_CODES,
)

_log = logging.getLogger(__name__)


def _parse_ollama_response(result: dict[str, Any]) -> str | None:
    """Extract and clean response text from Ollama JSON result."""
    if "error" in result:
        _log.warning("call_ollama API error: %s", result["error"])
        return None
    text = result.get("response", "").strip()
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
    text = text.strip("\"'").strip()
    return text if text else None


def call_ollama(
    queue_url: str,
    model: str,
    prompt: str,
    settings: dict[str, Any],
    timeout: int = 600,
    priority: int | None = None,
    source: str | None = None,
) -> str | None:
    """Call Ollama via queue and return cleaned response text.

    Retries up to _MAX_RETRIES times on 502/503 (model swap transients).
    Returns None on any error (network, timeout, parse).
    Strips <think>...</think> reasoning blocks from response.

    When priority/source are set, passes _priority/_source/_timeout to
    ollama-queue's proxy endpoint for job tracking and prioritization.
    """
    options = {}
    if "temperature" in settings:
        options["temperature"] = settings["temperature"]
    if "num_ctx" in settings:
        options["num_ctx"] = settings["num_ctx"]

    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        **({"options": options} if options else {}),
    }
    if priority is not None:
        body["_priority"] = priority
    if source is not None:
        body["_source"] = source
    if priority is not None or source is not None:
        body["_timeout"] = timeout

    payload = _json.dumps(body).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(  # noqa: S310
                f"{queue_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                result = _json.loads(resp.read().decode("utf-8"))
            return _parse_ollama_response(result)
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in _RETRYABLE_CODES and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                _log.warning("call_ollama %d retry in %.0fs: %s", exc.code, delay, exc)
                time.sleep(delay)
                continue
            _log.warning("call_ollama error: %s", exc)
            return None
        except (urllib.error.URLError, OSError, _json.JSONDecodeError) as exc:
            _log.warning("call_ollama error: %s", exc)
            return None

    _log.warning("call_ollama exhausted retries: %s", last_exc)
    return None


def _clean_principle(text: str) -> str:
    """Strip Chain-of-Thought artifacts from a generated principle.

    deepseek-r1 often includes reasoning traces, lesson-by-lesson analysis,
    and "This principle applies because..." explanations.  The judge should
    score the principle statement alone, not the surrounding rationale.
    """
    if not text:
        return text

    text = text.strip()

    # 1. If text starts with CoT preamble, try to find actual principle below
    cot_start = _re.match(
        r"^(okay|let me|let's|the lessons|here's|i'll|to analyze|looking at)",
        text,
        _re.IGNORECASE,
    )
    if cot_start:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs[1:]:
            # Skip paragraphs that are bullet lists or continuations of analysis
            if para.startswith("*") or para.startswith("-"):
                continue
            if len(para) > 20:
                text = para
                break

    # 2. Extract text after "**Principle:**" or "The principle is:" markers
    marker = _re.search(
        r"(?:\*\*Principle:\*\*|The principle is:)\s*(.+?)(?:\n\n|$)",
        text,
        _re.DOTALL,
    )
    if marker:
        text = marker.group(1).strip()

    # 3. Take only the first paragraph (strip trailing explanations)
    if "\n\n" in text:
        text = text.split("\n\n")[0].strip()

    # 4. Strip markdown bold markers
    text = _re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    # 5. Strip trailing parenthetical explanations like "*(This principle...)"
    text = _re.sub(r"\s*\*?\(This principle\b.*", "", text, flags=_re.DOTALL)

    return text.strip()


def call_judge(
    prompt: str,
    backend: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
    priority: int | None = None,
) -> str | None:
    """Call the judge model and return raw response text.

    Routes to Ollama or OpenAI based on backend parameter.
    Returns None on any error.
    """
    if backend == "openai":
        return _call_openai(openai_api_key, openai_model, prompt)
    return call_ollama(ollama_url, ollama_model, prompt, {}, priority=priority, source="eval-judge")


def _call_openai(api_key: str, model: str, prompt: str) -> str | None:
    """Call OpenAI Chat Completions API. Returns response text or None."""
    payload = _json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.1,
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(  # noqa: S310
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            result = _json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, _json.JSONDecodeError, KeyError, IndexError) as exc:
        _log.warning("_call_openai error: %s", exc)
        return None
