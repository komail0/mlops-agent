"""Configuration read from environment variables (and an optional local .env file)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env", environ: dict[str, str] | None = None) -> None:
    """Load KEY=VALUE lines into the environment without overriding existing values."""
    env = os.environ if environ is None else environ
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env.setdefault(key, value)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Config:
    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_min_interval_s: float = 0.0
    max_steps: int = 12
    tool_timeout_s: float = 15.0
    run_timeout_s: float = 180.0
    confirm_ttl_s: float = 300.0
    k8s_allowed_namespaces: tuple[str, ...] = ("mlops-agent",)
    k8s_context: str = ""
    allowed_dags: tuple[str, ...] = ("training_pipeline",)
    mlflow_tracking_uri: str = "http://localhost:5000"
    airflow_base_url: str = "http://localhost:8080"
    airflow_username: str = ""
    airflow_password: str = field(default="", repr=False)
    dvc_repo_path: str = "."
    model_max_age_days: int = 30
    min_f1: float = 0.8


def load_config(environ: Mapping[str, str] | None = None) -> Config:
    env = os.environ if environ is None else environ
    defaults = Config()

    def get(name: str, default: str) -> str:
        return env.get(name, default)

    return Config(
        llm_base_url=get("LLM_BASE_URL", defaults.llm_base_url),
        llm_api_key=get("LLM_API_KEY", defaults.llm_api_key),
        llm_model=get("LLM_MODEL", defaults.llm_model),
        llm_min_interval_s=float(get("LLM_MIN_INTERVAL_SECONDS", str(defaults.llm_min_interval_s))),
        max_steps=int(get("MAX_STEPS", str(defaults.max_steps))),
        tool_timeout_s=float(get("TOOL_TIMEOUT_S", str(defaults.tool_timeout_s))),
        run_timeout_s=float(get("RUN_TIMEOUT_S", str(defaults.run_timeout_s))),
        confirm_ttl_s=float(get("CONFIRM_TTL_S", str(defaults.confirm_ttl_s))),
        k8s_allowed_namespaces=tuple(
            _csv(get("K8S_ALLOWED_NAMESPACES", ",".join(defaults.k8s_allowed_namespaces)))
        ),
        k8s_context=get("K8S_CONTEXT", defaults.k8s_context),
        allowed_dags=tuple(_csv(get("ALLOWED_DAGS", ",".join(defaults.allowed_dags)))),
        mlflow_tracking_uri=get("MLFLOW_TRACKING_URI", defaults.mlflow_tracking_uri),
        airflow_base_url=get("AIRFLOW_BASE_URL", defaults.airflow_base_url),
        airflow_username=get("AIRFLOW_USERNAME", defaults.airflow_username),
        airflow_password=get("AIRFLOW_PASSWORD", defaults.airflow_password),
        dvc_repo_path=get("DVC_REPO_PATH", defaults.dvc_repo_path),
        model_max_age_days=int(get("MODEL_MAX_AGE_DAYS", str(defaults.model_max_age_days))),
        min_f1=float(get("MIN_F1", str(defaults.min_f1))),
    )
