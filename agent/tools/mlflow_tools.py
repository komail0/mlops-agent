"""MLflow tools built on MlflowClient (no scraping of the web UI).

Each tool returns recorded values only. Selection tools also return the facts a reader needs
to judge a candidate (dataset version, other metrics, sample sizes, quality-gate threshold,
age), because the best score alone does not make a model production-ready.
"""

from __future__ import annotations

import os
import time
from typing import Any

from agent.config import Config
from agent.schemas import ToolError, ToolSpec, object_schema

DATASET_TAGS = (
    "dataset.version",
    "dvc.md5",
    "dvc.dataset_commit",
    "dvc.dataset_path",
    "git.commit",
    "dvc.uncommitted_change",
    "dvc.workspace_matches",
)
READINESS_NOTE = (
    "The best metric value alone does not show a model is production-ready. Check the "
    "caveats: dataset version, other metrics, test-set size, quality-gate status and age."
)


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000))


def _is_missing(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_does_not_exist" in text or "does not exist" in text or "not found" in text


class MlflowTools:
    def __init__(self, config: Config, client: Any = None, now_ms: Any = None):
        self.config = config
        self._client = client
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        # Fail fast when the server is down instead of retrying for minutes.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(int(config.tool_timeout_s)))

    @property
    def client(self):
        if self._client is None:
            from mlflow import MlflowClient

            self._client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)
        return self._client

    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # network, auth or API errors become tool errors
            if _is_missing(exc):
                raise ToolError(f"not found: {exc}") from None
            raise ToolError(f"MLflow unavailable or request failed: {type(exc).__name__}") from None

    def _experiment_id(self, experiment: str) -> str:
        found = self._call(self.client.get_experiment_by_name, experiment)
        if found is not None:
            return found.experiment_id
        if str(experiment).isdigit():
            try:
                return self._call(self.client.get_experiment, str(experiment)).experiment_id
            except ToolError:
                pass
        names = [e.name for e in self._call(self.client.search_experiments)]
        raise ToolError(f"experiment '{experiment}' not found; available: {names}")

    @staticmethod
    def _summary(run) -> dict[str, Any]:
        tags = run.data.tags
        return {
            "run_id": run.info.run_id,
            "name": tags.get("mlflow.runName"),
            "status": run.info.status,
            "started_at": _iso(run.info.start_time),
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "dataset": {k: tags[k] for k in DATASET_TAGS if k in tags},
        }

    def _runs(self, experiment: str, limit: int, order_by: list[str], filter_string: str = ""):
        exp_id = self._experiment_id(experiment)
        return self._call(
            self.client.search_runs,
            [exp_id],
            filter_string=filter_string,
            max_results=max(1, min(int(limit), 200)),
            order_by=order_by,
        )

    # ---- tools --------------------------------------------------------------

    def list_experiments(self) -> dict[str, Any]:
        exps = self._call(self.client.search_experiments)
        return {
            "experiments": [
                {
                    "experiment_id": e.experiment_id,
                    "name": e.name,
                    "lifecycle_stage": e.lifecycle_stage,
                    "created_at": _iso(e.creation_time),
                }
                for e in exps
            ],
            "count": len(exps),
        }

    def list_runs(self, experiment: str, limit: int = 10) -> dict[str, Any]:
        runs = self._runs(experiment, limit, ["attributes.start_time DESC"])
        return {"experiment": experiment, "runs": [self._summary(r) for r in runs]}

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self._call(self.client.get_run, run_id)
        artifacts = self._call(self.client.list_artifacts, run_id)
        info = self._summary(run)
        info["tags"] = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
        info["artifacts"] = [
            {"path": a.path, "is_dir": a.is_dir, "size": a.file_size} for a in artifacts
        ]
        info["artifact_uri"] = run.info.artifact_uri
        return info

    def best_run(
        self, experiment: str, metric: str, mode: str = "max", limit: int = 50
    ) -> dict[str, Any]:
        if mode not in ("max", "min"):
            raise ToolError("mode must be 'max' or 'min'")
        runs = self._runs(
            experiment, limit, ["attributes.start_time DESC"], "attributes.status = 'FINISHED'"
        )
        scored = [r for r in runs if metric in r.data.metrics]
        missing = len(runs) - len(scored)
        if not scored:
            raise ToolError(
                f"no finished runs in '{experiment}' record the metric '{metric}' "
                f"({len(runs)} finished runs examined)"
            )
        sign = 1 if mode == "max" else -1
        ranked = sorted(scored, key=lambda r: sign * r.data.metrics[metric], reverse=True)
        best_value = ranked[0].data.metrics[metric]
        tied = [r for r in ranked if r.data.metrics[metric] == best_value]
        best = ranked[0]
        summary = self._summary(best)
        versions = {r.data.tags.get("dataset.version", "unknown") for r in scored}
        age_days = (self._now_ms() - (best.info.start_time or 0)) / 86_400_000
        caveats: dict[str, Any] = {
            "runs_considered": len(scored),
            "runs_missing_metric": missing,
            "tied_with_best": [r.info.run_id for r in tied[1:]],
            "dataset_versions_among_candidates": sorted(versions),
            "candidates_use_different_datasets": len(versions) > 1,
            "best_run_age_days": round(age_days, 1),
            "freshness_policy_max_age_days": self.config.model_max_age_days,
            "older_than_freshness_policy": age_days > self.config.model_max_age_days,
            "n_test": summary["metrics"].get("n_test"),
        }
        if metric == "f1":
            caveats["quality_gate_min_f1"] = self.config.min_f1
            caveats["passes_quality_gate"] = best_value >= self.config.min_f1
        return {
            "metric": metric,
            "mode": mode,
            "best_run": summary,
            "ranking": [
                {
                    "run_id": r.info.run_id,
                    "value": r.data.metrics[metric],
                    "dataset_version": r.data.tags.get("dataset.version"),
                }
                for r in ranked[:5]
            ],
            "caveats": caveats,
            "note": READINESS_NOTE,
        }

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        if not isinstance(run_ids, list) or len(run_ids) < 2:
            raise ToolError("provide at least two run ids")
        rows = [self._summary(self._call(self.client.get_run, rid)) for rid in run_ids[:10]]
        metric_names = sorted({m for r in rows for m in r["metrics"]})
        param_names = sorted({p for r in rows for p in r["params"]})
        return {
            "runs": rows,
            "metrics": {m: {r["run_id"]: r["metrics"].get(m) for r in rows} for m in metric_names},
            "params_that_differ": {
                p: {r["run_id"]: r["params"].get(p) for r in rows}
                for p in param_names
                if len({r["params"].get(p) for r in rows}) > 1
            },
            "datasets": {r["run_id"]: r["dataset"].get("dataset.version") for r in rows},
        }

    def compare_experiments(
        self, experiments: list[str], metric: str, mode: str = "max"
    ) -> dict[str, Any]:
        if not isinstance(experiments, list) or len(experiments) < 2:
            raise ToolError("provide at least two experiments")
        sign = 1 if mode == "max" else -1
        out = {}
        for name in experiments[:5]:
            runs = self._runs(name, 100, ["attributes.start_time DESC"])
            values = [r.data.metrics[metric] for r in runs if metric in r.data.metrics]
            best = (
                max(
                    (r for r in runs if metric in r.data.metrics),
                    key=lambda r: sign * r.data.metrics[metric],
                )
                if values
                else None
            )
            out[name] = {
                "runs": len(runs),
                "runs_with_metric": len(values),
                "best_value": best.data.metrics[metric] if best else None,
                "best_run_id": best.info.run_id if best else None,
                "mean_value": round(sum(values) / len(values), 6) if values else None,
            }
        return {"metric": metric, "mode": mode, "experiments": out}

    def registry_info(self, model_name: str | None = None) -> dict[str, Any]:
        try:
            models = self._call(self.client.search_registered_models)
        except ToolError:
            raise
        if model_name:
            models = [m for m in models if m.name == model_name]
            if not models:
                return {"available": False, "reason": f"model '{model_name}' is not registered"}
        if not models:
            return {"available": False, "reason": "no registered models exist"}
        result = []
        for m in models:
            versions = self._call(
                self.client.search_model_versions, f"name='{m.name}'", max_results=20
            )
            result.append(
                {
                    "name": m.name,
                    "aliases": dict(getattr(m, "aliases", {}) or {}),
                    "versions": [
                        {
                            "version": v.version,
                            "run_id": v.run_id,
                            "status": v.status,
                            "aliases": list(getattr(v, "aliases", []) or []),
                            "created_at": _iso(v.creation_timestamp),
                        }
                        for v in versions
                    ],
                }
            )
        return {"available": True, "models": result}

    # ---- registry -----------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        exp = {"type": "string", "description": "Experiment name or id."}
        metric = {"type": "string", "description": "Metric name, e.g. f1, accuracy."}
        mode = {"type": "string", "enum": ["max", "min"], "description": "max if higher is better."}
        return [
            ToolSpec(
                "mlflow_list_experiments",
                "List MLflow experiments.",
                object_schema(),
                self.list_experiments,
            ),
            ToolSpec(
                "mlflow_list_runs",
                "List recent runs of an experiment, newest first, with metrics, params and the "
                "recorded dataset version.",
                object_schema(
                    {"experiment": exp, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                    ["experiment"],
                ),
                self.list_runs,
            ),
            ToolSpec(
                "mlflow_get_run",
                "Get one run: metrics, parameters, tags, dataset provenance and artifacts.",
                object_schema({"run_id": {"type": "string"}}, ["run_id"]),
                self.get_run,
            ),
            ToolSpec(
                "mlflow_best_run",
                "Find the best finished run by a metric, with caveats (dataset versions, ties, "
                "age, quality gate). Does not decide production readiness.",
                object_schema(
                    {
                        "experiment": exp,
                        "metric": metric,
                        "mode": mode,
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    ["experiment", "metric"],
                ),
                self.best_run,
            ),
            ToolSpec(
                "mlflow_compare_runs",
                "Compare two or more runs side by side: metrics, differing params, datasets.",
                object_schema(
                    {"run_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
                    ["run_ids"],
                ),
                self.compare_runs,
            ),
            ToolSpec(
                "mlflow_compare_experiments",
                "Compare two or more experiments on one metric (best, mean, run counts).",
                object_schema(
                    {
                        "experiments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                        "metric": metric,
                        "mode": mode,
                    },
                    ["experiments", "metric"],
                ),
                self.compare_experiments,
            ),
            ToolSpec(
                "mlflow_registry_info",
                "Get model registry information (versions, aliases) if a registry exists.",
                object_schema({"model_name": {"type": "string"}}),
                self.registry_info,
            ),
        ]
