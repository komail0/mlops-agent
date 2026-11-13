"""Thin wrapper over an OpenAI-compatible chat API with tool calling.

It only sends messages and parses the reply. Choosing tools and using their results is the
orchestrator's job. Retries handle rate limits and transient network errors.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import OpenAI

from agent.config import Config
from agent.redact import redact

REQUEST_TIMEOUT_S = 90
RETRY_DELAYS_S = (5, 15, 30, 60)
MAX_RETRY_WAIT_S = 90  # a longer server-requested wait (e.g. a daily quota) is not worth waiting
ERROR_DETAIL_CHARS = 500
_RETRY_HINT = re.compile(
    r"retry(?:\s*delay)?['\"]?\s*(?:in|:)\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE
)
_RETRIABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class LLMError(Exception):
    """The model could not be reached or rejected the request."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None
    # Provider metadata that must be echoed back with the call (e.g. Gemini thought signatures).
    extra_content: dict[str, Any] | None = None


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    def as_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            calls = []
            for call in self.tool_calls:
                entry: dict[str, Any] = {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                if call.extra_content:
                    entry["extra_content"] = call.extra_content
                calls.append(entry)
            message["tool_calls"] = calls
        return message


def _retry_hint(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    try:
        if header is not None:
            return float(header)
    except ValueError:
        pass
    match = _RETRY_HINT.search(str(exc))
    return float(match.group(1)) if match else None


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}, "tool arguments were not valid JSON"
    if not isinstance(parsed, dict):
        return {}, "tool arguments must be a JSON object"
    return parsed, None


class LLMClient:
    def __init__(
        self,
        config: Config,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        retry_delays: Sequence[float] = RETRY_DELAYS_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._config = config
        self._client = client or OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url or None,
            timeout=REQUEST_TIMEOUT_S,
            max_retries=0,
        )
        self._sleep = sleep
        self._retry_delays = tuple(retry_delays)
        self._clock = clock
        self._last_request: float | None = None

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None):
        kwargs: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        self._space_requests()
        return self._parse(self._request(kwargs))

    def _space_requests(self) -> None:
        interval = self._config.llm_min_interval_s
        now = self._clock()
        if interval > 0 and self._last_request is not None:
            wait = self._last_request + interval - now
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_request = now

    def _request(self, kwargs: dict[str, Any]) -> Any:
        attempts = len(self._retry_delays) + 1
        for attempt in range(attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except _RETRIABLE as exc:
                if attempt == attempts - 1:
                    raise LLMError(
                        f"LLM request failed after {attempts} attempts: {self._describe(exc)}"
                    ) from None
                delay = self._retry_delays[attempt]
                hint = _retry_hint(exc) if isinstance(exc, openai.RateLimitError) else None
                if hint is not None:
                    if hint > MAX_RETRY_WAIT_S:
                        raise LLMError(
                            f"rate limited; the server asks to wait {hint:.0f}s: "
                            f"{self._describe(exc)}"
                        ) from None
                    delay = hint + 1
                self._sleep(delay)
            except openai.OpenAIError as exc:
                raise LLMError(f"LLM request rejected: {self._describe(exc)}") from None
        raise LLMError("LLM request failed")

    def _describe(self, exc: Exception) -> str:
        status = getattr(exc, "status_code", None)
        detail = redact(str(getattr(exc, "message", None) or exc))[:ERROR_DETAIL_CHARS]
        return f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "") + f": {detail}"

    @staticmethod
    def _parse(response: Any) -> LLMReply:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return LLMReply(content="")
        choice = choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for index, raw_call in enumerate(getattr(message, "tool_calls", None) or []):
            function = raw_call.function
            arguments, error = _parse_arguments(getattr(function, "arguments", None))
            extra = getattr(raw_call, "extra_content", None)
            calls.append(
                ToolCall(
                    id=getattr(raw_call, "id", None) or f"call_{index}",
                    name=getattr(function, "name", "") or "",
                    arguments=arguments,
                    parse_error=error,
                    extra_content=extra if isinstance(extra, dict) else None,
                )
            )
        return LLMReply(
            content=getattr(message, "content", None) or "",
            tool_calls=calls,
            finish_reason=getattr(choice, "finish_reason", None),
        )
