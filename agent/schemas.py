"""Shared data structures: the tool result envelope and the tool specification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    """A tool failed in an expected way. The message is safe to show to the model."""


@dataclass
class ToolResult:
    """Uniform envelope returned by every tool. Failures are values, not exceptions."""

    ok: bool
    tool: str
    data: Any = None
    error: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def success(cls, tool: str, data: Any) -> ToolResult:
        return cls(ok=True, tool=tool, data=data)

    @classmethod
    def failure(cls, tool: str, error: str) -> ToolResult:
        return cls(ok=False, tool=tool, error=error)


@dataclass
class ToolSpec:
    """A callable tool with the JSON schema that is sent to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    mutating: bool = False
    timeout_s: float | None = None  # overrides the default per-tool time limit
    tags: list[str] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def object_schema(
    properties: dict[str, Any] | None = None, required: list[str] | None = None
) -> dict[str, Any]:
    """Build a strict JSON-schema object for tool parameters."""
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }
