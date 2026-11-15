"""Required tool coverage, checked in code before an answer is accepted.

The prompt asks the model to look at every relevant system, but a model sometimes stops early.
For questions where the brief requires several systems, the orchestrator therefore checks which
tools were called, and sends the model back for the ones that are missing. A tool that was
called and failed still counts: an unreachable system must not trap the loop.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

RETRAIN_RE = re.compile(r"re-?train|new model|model (?:is |has )?(?:stale|outdated|degrad)", re.I)
FIRST_STATE_RE = re.compile(r'"state":\s*"(\w+)"')


class _Record(Protocol):
    name: str
    ok: bool
    output: str


def missing_coverage(user_text: str, records: Iterable[_Record]) -> list[str]:
    """Return what must still be checked before this question can be answered."""
    if not RETRAIN_RE.search(user_text):
        return []
    records = list(records)
    called = {r.name for r in records}
    missing: list[str] = []
    if not any(name.startswith("mlflow_") for name in called):
        missing.append("MLflow, for the best run and its caveats (for example mlflow_best_run)")
    if not any(name.startswith("dvc_") for name in called):
        missing.append("DVC, for the dataset state (for example dvc_dataset_version)")
    if not called & {"airflow_dag_runs", "airflow_run_detail"}:
        missing.append("Airflow, for the most recent pipeline runs (airflow_dag_runs)")
    if not called & {"k8s_deployed_version", "k8s_list_deployments"}:
        missing.append("Kubernetes, for the running version (k8s_deployed_version)")
    if "airflow_failed_tasks" not in called:
        for record in records:
            if record.name == "airflow_dag_runs" and record.ok:
                first = FIRST_STATE_RE.search(record.output)
                if first and first.group(1) == "failed":
                    missing.append("airflow_failed_tasks, because the latest pipeline run failed")
    return missing
