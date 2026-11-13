"""Safety guards enforced in code, never only in the prompt.

* Allowlists for Kubernetes namespaces, Airflow DAGs and DVC subcommands.
* A confirmation manager for state-changing operations. The model can request an
  action, but only `approve()` (called from the user-input path) releases it.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

DVC_ALLOWED_SUBCOMMANDS = frozenset({"list", "status", "diff", "checkout", "pull"})
_FORBIDDEN_CHARS = re.compile(r"[;&|`$<>\n\r\\]|\$\(")
_SAFE_ARG = re.compile(r"^[A-Za-z0-9_./:@=+-]+$")


class SafetyError(Exception):
    """Raised when a request violates an allowlist or the confirmation rules."""


def check_namespace(namespace: str, allowed: Iterable[str]) -> str:
    if namespace not in set(allowed):
        raise SafetyError(f"namespace '{namespace}' is not approved")
    return namespace


def check_dag(dag_id: str, allowed: Iterable[str]) -> str:
    if dag_id not in set(allowed):
        raise SafetyError(f"DAG '{dag_id}' is not approved")
    return dag_id


def validate_dvc_args(args: Sequence[str]) -> list[str]:
    """Validate a DVC argument list. Only approved subcommands and safe arguments pass."""
    if not args:
        raise SafetyError("empty DVC command")
    items = [str(a) for a in args]
    if items[0] not in DVC_ALLOWED_SUBCOMMANDS:
        raise SafetyError(f"DVC subcommand '{items[0]}' is not approved")
    for arg in items[1:]:
        if _FORBIDDEN_CHARS.search(arg):
            raise SafetyError("DVC argument contains forbidden characters")
        if arg.startswith("-"):
            if not _SAFE_ARG.match(arg):
                raise SafetyError("DVC option contains forbidden characters")
            continue
        if not _SAFE_ARG.match(arg):
            raise SafetyError("DVC argument contains forbidden characters")
        if (
            ".." in PurePosixPath(arg).parts
            or PurePosixPath(arg).is_absolute()
            or PureWindowsPath(arg).is_absolute()
            or re.match(r"^[A-Za-z]:", arg)
        ):
            raise SafetyError("DVC path must stay inside the project")
    return items


@dataclass
class _Pending:
    action: str
    params_key: str
    description: str
    expires_at: float
    approved: bool = False


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


class ConfirmationManager:
    """Two-step confirmation for state-changing operations."""

    def __init__(self, ttl_s: float = 300.0, clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl_s
        self._clock = clock
        self._pending: dict[str, _Pending] = {}

    def request(self, action: str, params: dict, description: str) -> str:
        """Called by a tool on behalf of the model. Returns a token; nothing runs yet."""
        token = secrets.token_hex(4)
        self._pending[token] = _Pending(
            action, _key(params), description, self._clock() + self._ttl
        )
        return token

    def describe(self, token: str) -> str | None:
        pending = self._pending.get(token)
        return pending.description if pending else None

    def approve(self, token: str) -> bool:
        """Called only from the user-input path. The model has no route to this method."""
        pending = self._pending.get(token)
        if pending is None or self._clock() > pending.expires_at:
            return False
        pending.approved = True
        return True

    def consume(self, token: str, action: str, params: dict) -> bool:
        """True only for an approved, unexpired token that matches the exact action and
        parameters. A token works once."""
        pending = self._pending.get(token)
        if pending is None:
            return False
        if self._clock() > pending.expires_at:
            del self._pending[token]
            return False
        if not pending.approved or pending.action != action or pending.params_key != _key(params):
            return False
        del self._pending[token]
        return True
