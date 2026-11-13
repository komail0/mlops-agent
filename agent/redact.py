"""Secret redaction applied to logs and to every tool output sent to the model."""

from __future__ import annotations

import re
from typing import Any

MASK = "[REDACTED]"

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),  # JWT
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.I),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"\bgsk_[0-9A-Za-z]{20,}"),
    re.compile(r"\bsk-[0-9A-Za-z_-]{16,}"),
    re.compile(r"\bgh[pousr]_[0-9A-Za-z]{20,}"),
]
_KEY_VALUE = re.compile(
    r"""(?ix)
    (["']?(?:[\w.-]*(?:password|passwd|secret|token|api[_-]?key|authorization)[\w.-]*)["']?
     \s*[:=]\s*)
    (["']?)([^\s"',;}]+)
    """
)
_SENSITIVE_KEY = re.compile(r"(?i)password|passwd|secret|token|api[_-]?key|authorization")


def redact(text: str) -> str:
    """Mask secrets in free text."""
    for pattern in _PATTERNS:
        text = pattern.sub(MASK, text)
    return _KEY_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}{MASK}", text)


def redact_obj(value: Any) -> Any:
    """Mask secrets in nested dicts, lists and strings."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {
            k: (MASK if isinstance(k, str) and _SENSITIVE_KEY.search(k) else redact_obj(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_obj(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_obj(v) for v in value)
    return value
