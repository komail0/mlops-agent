"""Structured JSON logging. Every record is redacted before it is written."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from typing import Any

from agent.redact import redact, redact_obj

LOGGER_NAME = "mlops_agent"
_EXTRA_FIELDS = ("request_id", "tool", "status", "duration_ms", "error", "detail")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "event": redact(record.getMessage()),
        }
        for name in _EXTRA_FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = redact_obj(value)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO, stream: Any = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_tool_call(
    request_id: str,
    tool: str,
    status: str,
    duration_ms: int,
    error: str | None = None,
) -> None:
    """Record one tool call: name, status, duration and request id. Never the arguments."""
    logging.getLogger(LOGGER_NAME).info(
        "tool_call",
        extra={
            "request_id": request_id,
            "tool": tool,
            "status": status,
            "duration_ms": duration_ms,
            "error": error,
        },
    )
