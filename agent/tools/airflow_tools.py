"""Airflow 3 tools using the authenticated REST API (`/api/v2`, JWT from `/auth/token`).

Triggering a DAG changes state, so it needs an allowlisted DAG id, an allowlisted set of
conf keys and the user's confirmation.
"""

from __future__ import annotations

import re
import time
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx

from agent.config import Config
from agent.redact import redact
from agent.safety import ConfirmationManager, SafetyError, check_dag
from agent.schemas import ToolError, ToolSpec, object_schema

_ID = re.compile(r"^[A-Za-z0-9_.:+-]{1,250}$")
TRIGGER_CONF_KEYS = {"data_path", "model_type", "seed", "min_f1"}
FAILED_STATES = {"failed", "upstream_failed"}
MAX_LOG_EVENTS = 40


def _check_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise ToolError(f"invalid {what}")
    return value


def _seg(value: str) -> str:
    return quote(value, safe="")


class AirflowClient:
    """Small authenticated client. Fetches a JWT, reuses it, refreshes it once on a 401."""

    def __init__(self, config: Config, transport: httpx.BaseTransport | None = None):
        self._base = config.airflow_base_url.rstrip("/")
        self._user = config.airflow_username
        self._password = config.airflow_password
        self._http = httpx.Client(timeout=config.tool_timeout_s, transport=transport)
        self._token: str | None = None
        self._token_at = 0.0

    def _login(self) -> str:
        if not self._user or not self._password:
            raise ToolError("Airflow credentials are not configured")
        resp = self._send(
            "POST", "/auth/token", json={"username": self._user, "password": self._password}
        )
        if resp.status_code in (401, 403):
            raise ToolError("Airflow authentication failed")
        self._raise_for_status(resp)
        self._token = resp.json()["access_token"]
        self._token_at = time.monotonic()
        return self._token

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, self._base + path, **kwargs)
        except httpx.TimeoutException:
            raise ToolError("Airflow request timed out") from None
        except httpx.HTTPError as exc:
            raise ToolError(f"Airflow unavailable: {type(exc).__name__}") from None

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code == 404:
            raise ToolError("not found in Airflow")
        if resp.status_code in (401, 403):
            raise ToolError("Airflow authentication failed")
        if resp.status_code >= 400:
            raise ToolError(f"Airflow returned HTTP {resp.status_code}")

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        for attempt in (1, 2):
            token = self._token or self._login()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            resp = self._send(method, path, headers=headers, **kwargs)
            if resp.status_code == 401 and attempt == 1:
                self._token = None  # expired token: log in again once
                continue
            self._raise_for_status(resp)
            try:
                return resp.json()
            except ValueError:
                raise ToolError("Airflow returned a non-JSON response") from None
        raise ToolError("Airflow authentication failed")


def _run_summary(run: dict) -> dict[str, Any]:
    return {
        "dag_run_id": run.get("dag_run_id"),
        "state": run.get("state"),
        "run_type": run.get("run_type"),
        "queued_at": run.get("queued_at"),
        "start_date": run.get("start_date"),
        "end_date": run.get("end_date"),
        "duration_s": run.get("duration"),
        "conf": run.get("conf") or {},
        "triggered_by": run.get("triggered_by"),
    }


def _task_summary(ti: dict) -> dict[str, Any]:
    return {
        "task_id": ti.get("task_id"),
        "state": ti.get("state"),
        "try_number": ti.get("try_number"),
        "max_tries": ti.get("max_tries"),
        "start_date": ti.get("start_date"),
        "end_date": ti.get("end_date"),
        "duration_s": ti.get("duration"),
    }


class AirflowTools:
    def __init__(
        self,
        config: Config,
        confirmations: ConfirmationManager,
        client: AirflowClient | None = None,
    ):
        self.config = config
        self.confirmations = confirmations
        self.client = client or AirflowClient(config)

    def _get(self, path: str, **params: Any) -> Any:
        return self.client.request("GET", path, params=params or None)

    # ---- tools --------------------------------------------------------------

    def list_dags(self) -> dict[str, Any]:
        data = self._get("/api/v2/dags", limit=50)
        dags = [
            {
                "dag_id": d["dag_id"],
                "is_paused": d.get("is_paused"),
                "tags": [t["name"] for t in d.get("tags", [])],
                "approved_for_trigger": d["dag_id"] in self.config.allowed_dags,
            }
            for d in data.get("dags", [])
        ]
        return {"dags": dags, "count": len(dags)}

    def dag_runs(self, dag_id: str, limit: int = 5) -> dict[str, Any]:
        _check_id(dag_id, "dag_id")
        limit = max(1, min(int(limit), 25))
        data = self._get(
            f"/api/v2/dags/{_seg(dag_id)}/dagRuns", order_by="-start_date", limit=limit
        )
        return {"dag_id": dag_id, "runs": [_run_summary(r) for r in data.get("dag_runs", [])]}

    def _latest_run_id(self, dag_id: str, failed_only: bool = False) -> str:
        params: dict[str, Any] = {"order_by": "-start_date", "limit": 1}
        if failed_only:
            params["state"] = "failed"
        runs = self._get(f"/api/v2/dags/{_seg(dag_id)}/dagRuns", **params).get("dag_runs", [])
        if not runs:
            kind = "failed " if failed_only else ""
            raise ToolError(f"no {kind}runs found for DAG '{dag_id}'")
        return runs[0]["dag_run_id"]

    def run_detail(self, dag_id: str, run_id: str | None = None) -> dict[str, Any]:
        _check_id(dag_id, "dag_id")
        run_id = _check_id(run_id, "run_id") if run_id else self._latest_run_id(dag_id)
        run = self._get(f"/api/v2/dags/{_seg(dag_id)}/dagRuns/{_seg(run_id)}")
        tis = self._get(f"/api/v2/dags/{_seg(dag_id)}/dagRuns/{_seg(run_id)}/taskInstances").get(
            "task_instances", []
        )
        return {
            "dag_id": dag_id,
            "run": _run_summary(run),
            "tasks": [_task_summary(t) for t in tis],
        }

    def failed_tasks(self, dag_id: str, run_id: str | None = None) -> dict[str, Any]:
        _check_id(dag_id, "dag_id")
        run_id = _check_id(run_id, "run_id") if run_id else self._latest_run_id(dag_id, True)
        tis = self._get(f"/api/v2/dags/{_seg(dag_id)}/dagRuns/{_seg(run_id)}/taskInstances").get(
            "task_instances", []
        )
        failed = [_task_summary(t) for t in tis if t.get("state") in FAILED_STATES]
        return {
            "dag_id": dag_id,
            "dag_run_id": run_id,
            "failed_tasks": [t for t in failed if t["state"] == "failed"],
            "blocked_downstream_tasks": [t for t in failed if t["state"] == "upstream_failed"],
            "note": "'failed' tasks are root failures; 'upstream_failed' tasks never ran "
            "because an upstream task failed.",
        }

    def task_logs(
        self, dag_id: str, run_id: str, task_id: str, try_number: int | None = None
    ) -> dict[str, Any]:
        _check_id(dag_id, "dag_id")
        _check_id(run_id, "run_id")
        _check_id(task_id, "task_id")
        if try_number is None:
            ti = self._get(
                f"/api/v2/dags/{_seg(dag_id)}/dagRuns/{_seg(run_id)}/taskInstances/{_seg(task_id)}"
            )
            try_number = int(ti.get("try_number") or 1)
        data = self._get(
            f"/api/v2/dags/{_seg(dag_id)}/dagRuns/{_seg(run_id)}/taskInstances/"
            f"{_seg(task_id)}/logs/{int(try_number)}"
        )
        events = data.get("content", [])
        lines, errors = [], []
        for ev in events:
            if isinstance(ev, str):
                lines.append(redact(ev)[:300])
                continue
            line = f"{ev.get('timestamp', '')} {ev.get('level', '')} {ev.get('event', '')}"
            lines.append(redact(line.strip())[:300])
            for detail in ev.get("error_detail") or []:
                errors.append(redact(f"{detail.get('exc_type')}: {detail.get('exc_value')}")[:400])
        return {
            "dag_id": dag_id,
            "dag_run_id": run_id,
            "task_id": task_id,
            "try_number": int(try_number),
            "errors": errors,
            "log_tail": lines[-MAX_LOG_EVENTS:],
            "total_events": len(events),
        }

    def trigger_dag(
        self,
        dag_id: str,
        conf: dict | None = None,
        confirmation_token: str | None = None,
    ) -> dict[str, Any]:
        _check_id(dag_id, "dag_id")
        try:
            check_dag(dag_id, self.config.allowed_dags)
        except SafetyError as exc:
            raise ToolError(str(exc)) from None
        conf = self._check_conf(conf or {})
        params = {"dag_id": dag_id, "conf": conf}
        if not confirmation_token:
            description = f"Trigger DAG '{dag_id}' with conf {conf or '{}'}"
            token = self.confirmations.request("airflow_trigger_dag", params, description)
            raise ToolError(
                f"confirmation_required: ask the user to confirm token '{token}' for: "
                f"{description}. Then call again with confirmation_token."
            )
        if not self.confirmations.consume(confirmation_token, "airflow_trigger_dag", params):
            raise ToolError("confirmation token is invalid, unapproved, expired or mismatched")
        run = self.client.request(
            "POST",
            f"/api/v2/dags/{_seg(dag_id)}/dagRuns",
            json={"logical_date": None, "conf": conf},
        )
        return {"triggered": True, "dag_id": dag_id, **_run_summary(run)}

    @staticmethod
    def _check_conf(conf: dict) -> dict:
        extra = set(conf) - TRIGGER_CONF_KEYS
        if extra:
            raise ToolError(
                f"conf keys not allowed: {sorted(extra)}; allowed: {sorted(TRIGGER_CONF_KEYS)}"
            )
        path = conf.get("data_path")
        if path is not None:
            p = PurePosixPath(str(path).replace("\\", "/"))
            if p.is_absolute() or ".." in p.parts or not re.match(r"^[\w./-]+$", str(path)):
                raise ToolError("data_path must be a relative path inside the project")
        if "model_type" in conf and conf["model_type"] not in ("logreg", "rf"):
            raise ToolError("model_type must be 'logreg' or 'rf'")
        return conf

    # ---- registry -----------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        dag = {"type": "string", "description": "DAG id, e.g. training_pipeline."}
        run = {"type": "string", "description": "DAG run id. Omit for the latest."}
        return [
            ToolSpec(
                "airflow_list_dags",
                "List Airflow DAGs and whether each may be triggered.",
                object_schema(),
                self.list_dags,
            ),
            ToolSpec(
                "airflow_dag_runs",
                "List recent runs of a DAG with state and timing, newest first.",
                object_schema(
                    {"dag_id": dag, "limit": {"type": "integer", "minimum": 1, "maximum": 25}},
                    ["dag_id"],
                ),
                self.dag_runs,
            ),
            ToolSpec(
                "airflow_run_detail",
                "Get one DAG run with the state of each task. Defaults to the latest run.",
                object_schema({"dag_id": dag, "run_id": run}, ["dag_id"]),
                self.run_detail,
            ),
            ToolSpec(
                "airflow_failed_tasks",
                "Identify failed tasks in a run (default: latest failed run), separating root "
                "failures from tasks blocked by them.",
                object_schema({"dag_id": dag, "run_id": run}, ["dag_id"]),
                self.failed_tasks,
            ),
            ToolSpec(
                "airflow_task_logs",
                "Get the log tail and error details of a task instance, for diagnosing failures.",
                object_schema(
                    {
                        "dag_id": dag,
                        "run_id": {"type": "string"},
                        "task_id": {"type": "string"},
                        "try_number": {"type": "integer", "minimum": 1},
                    },
                    ["dag_id", "run_id", "task_id"],
                ),
                self.task_logs,
            ),
            ToolSpec(
                "airflow_trigger_dag",
                "Trigger an approved DAG run. Starts work, so it needs the user's confirmation. "
                "Never use this to retrain unless the user explicitly asks for it.",
                object_schema(
                    {
                        "dag_id": dag,
                        "conf": {
                            "type": "object",
                            "description": "Optional: data_path, model_type, seed, min_f1.",
                        },
                        "confirmation_token": {
                            "type": "string",
                            "description": "Token the user approved. Omit on the first call.",
                        },
                    },
                    ["dag_id"],
                ),
                self.trigger_dag,
                mutating=True,
            ),
        ]
