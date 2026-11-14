"""The agent loop: the model picks tools, the orchestrator executes them and feeds results back.

Observe (user request and prior turns) -> reason and select tools (model) -> execute
(validated, time-limited, errors returned as values) -> interpret (model reads the results)
-> answer. State-changing tools are released only by the user's confirmation, handled here in
code: the model has no way to approve anything.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.config import Config
from agent.llm import LLMError, ToolCall
from agent.logging_setup import LOGGER_NAME, log_tool_call, new_request_id
from agent.prompts import CONFIRMED_MESSAGE, SYSTEM_PROMPT
from agent.redact import redact, redact_obj
from agent.safety import ConfirmationManager, SafetyError
from agent.schemas import ToolError, ToolResult, ToolSpec
from agent.validation import validate_args

MAX_TOOL_CHARS = 7000
EVIDENCE_CHARS = 1500
KEEP_TURNS = 4
CONFIRM_RE = re.compile(r"confirm\s+([0-9a-f]{8})", re.IGNORECASE)
UNAVAILABLE_NOTE = "Information unavailable. Do not estimate or invent it; say it is unavailable."
CONFIRM_NOTE = (
    "Nothing was changed. Tell the user to reply exactly `confirm <token>` using the token in "
    "the error, then stop. Do not call this tool again until they confirm."
)


@dataclass
class ToolRecord:
    name: str
    ok: bool
    duration_ms: int
    error: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    output: str = ""  # redacted, truncated tool output, kept as evidence for the report


@dataclass
class AgentResult:
    answer: str
    request_id: str
    status: str  # ok, step_limit, timeout, llm_error, rejected
    steps: int = 0
    tools: list[ToolRecord] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        config: Config,
        llm: Any,
        registry: dict[str, ToolSpec],
        confirmations: ConfirmationManager,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.llm = llm
        self.registry = registry
        self.confirmations = confirmations
        self._clock = clock
        self._turns: list[list[dict[str, Any]]] = []
        self._tool_schemas = [spec.to_openai() for spec in registry.values()]
        self._log = logging.getLogger(LOGGER_NAME)

    # ---- public -------------------------------------------------------------

    def chat(self, user_text: str) -> AgentResult:
        """Handle one user message. A `confirm <token>` message is approved here, in code."""
        request_id = new_request_id()
        match = CONFIRM_RE.fullmatch(user_text.strip())
        if match:
            token = match.group(1).lower()
            if not self.confirmations.approve(token):
                return AgentResult(
                    "That confirmation token is not valid or has expired. Nothing was changed.",
                    request_id,
                    "rejected",
                )
            user_text = CONFIRMED_MESSAGE.format(token=token)
        return self._run(request_id, user_text)

    # ---- loop ---------------------------------------------------------------

    def _run(self, request_id: str, user_text: str) -> AgentResult:
        self._log.info("request_start", extra={"request_id": request_id})
        started = self._clock()
        turn: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
        records: list[ToolRecord] = []
        status, answer, steps = "ok", "", 0

        def messages() -> list[dict[str, Any]]:
            history = [m for t in self._turns[-KEEP_TURNS:] for m in t]
            return [{"role": "system", "content": SYSTEM_PROMPT}, *history, *turn]

        try:
            for step in range(1, self.config.max_steps + 1):
                steps = step
                if self._clock() - started > self.config.run_timeout_s:
                    status = "timeout"
                    break
                reply = self.llm.chat(messages(), self._tool_schemas)
                turn.append(reply.as_message())
                if not reply.tool_calls:
                    answer = reply.content.strip()
                    break
                for call in reply.tool_calls:
                    result = self._execute(call, request_id)
                    rendered = self._render(result)
                    records.append(
                        ToolRecord(
                            result.tool,
                            result.ok,
                            result.duration_ms,
                            result.error,
                            redact_obj(call.arguments),
                            rendered[:EVIDENCE_CHARS],
                        )
                    )
                    turn.append({"role": "tool", "tool_call_id": call.id, "content": rendered})
            else:
                status = "step_limit"
            if status in ("timeout", "step_limit"):
                turn.append(
                    {
                        "role": "user",
                        "content": "The step or time limit was reached. Answer now using only the "
                        "evidence already gathered, and say what could not be checked.",
                    }
                )
                answer = self.llm.chat(messages(), None).content.strip()
                turn.append({"role": "assistant", "content": answer})
        except LLMError as exc:
            status = "llm_error"
            answer = (
                "I could not reach the language model, so I cannot answer or check anything "
                f"right now. Reason: {redact(str(exc))[:300]}"
            )
        if not answer:
            answer = "No answer was produced."
        if status != "llm_error":
            self._turns.append(turn)
        self._log.info(
            "request_end",
            extra={"request_id": request_id, "status": status, "detail": {"steps": steps}},
        )
        return AgentResult(answer, request_id, status, steps, records)

    # ---- tool execution -----------------------------------------------------

    def _execute(self, call: ToolCall, request_id: str) -> ToolResult:
        began = time.monotonic()
        result = self._dispatch(call)
        result.duration_ms = int((time.monotonic() - began) * 1000)
        log_tool_call(
            request_id,
            result.tool,
            "ok" if result.ok else "error",
            result.duration_ms,
            result.error,
        )
        return result

    def _dispatch(self, call: ToolCall) -> ToolResult:
        name = call.name or "(none)"
        spec = self.registry.get(call.name)
        if spec is None:
            return ToolResult.failure(name, f"unknown tool; available: {sorted(self.registry)}")
        if call.parse_error:
            return ToolResult.failure(name, f"invalid arguments: {call.parse_error}")
        problem = validate_args(spec.parameters, call.arguments)
        if problem:
            return ToolResult.failure(name, f"invalid arguments: {problem}")
        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                outcome["data"] = spec.handler(**call.arguments)
            except (ToolError, SafetyError) as exc:
                outcome["error"] = str(exc)
            except Exception as exc:  # never let a tool crash the loop
                outcome["error"] = f"internal error in tool ({type(exc).__name__})"

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        limit = spec.timeout_s or self.config.tool_timeout_s
        thread.join(limit)
        if thread.is_alive():
            return ToolResult.failure(name, f"timed out after {limit:g}s")
        if "error" in outcome:
            return ToolResult.failure(name, redact(outcome["error"]))
        return ToolResult.success(name, redact_obj(outcome.get("data")))

    @staticmethod
    def _render(result: ToolResult) -> str:
        payload = result.to_dict()
        if not result.ok:
            confirm = (result.error or "").startswith("confirmation_required")
            payload["instruction"] = CONFIRM_NOTE if confirm else UNAVAILABLE_NOTE
        text = json.dumps(payload, default=str)
        if len(text) > MAX_TOOL_CHARS:
            text = text[:MAX_TOOL_CHARS] + '... [truncated]"}'
        return text
