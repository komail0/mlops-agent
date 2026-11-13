"""Training DAG: validation -> preprocessing -> training -> evaluation/MLflow -> quality gate.

A failed validation or quality gate stops every downstream task, so nothing is registered
or promoted from bad data or a weak model. Stage logic lives in `training/pipeline.py`.

Trigger parameters (all optional): data_path, model_type (logreg|rf), seed, min_f1.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowFailException

log = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/project")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "mlops-agent-demo")
WORK_ROOT = os.environ.get("PIPELINE_WORK_DIR", "/tmp/ml_work")

default_args = {
    "owner": "mlops-agent",
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}


@dag(
    dag_id="training_pipeline",
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["ml", "training"],
    params={
        "data_path": Param("data/dataset.csv", type="string"),
        "model_type": Param("logreg", enum=["logreg", "rf"]),
        "seed": Param(42, type="integer"),
        "min_f1": Param(0.8, type="number"),
    },
)
def training_pipeline():
    @task
    def validate_data(params=None, run_id=None):
        from training.pipeline import ValidationError, validate_dataset

        path = os.path.join(PROJECT_ROOT, params["data_path"])
        log.info("validating dataset %s", params["data_path"])
        try:
            report = validate_dataset(path)
        except ValidationError as exc:
            # Bad data cannot be fixed by retrying, so fail immediately.
            log.error("validation failed: %s", exc)
            raise AirflowFailException(str(exc)) from None
        log.info("validation passed: %s", report)
        return {"path": path, **report}

    @task
    def preprocess(validated, params=None, run_id=None):
        from training.pipeline import preprocess as run_preprocess

        workdir = os.path.join(WORK_ROOT, run_id)
        result = run_preprocess(validated["path"], workdir, params["seed"])
        log.info("preprocessed: %s", result)
        return {**result, "data_path": validated["path"]}

    @task
    def train_model(prepared, params=None):
        from training.pipeline import train_model as run_train

        model_path = run_train(prepared["workdir"], params["model_type"], params["seed"])
        log.info("model saved to %s", model_path)
        return prepared

    @task
    def evaluate_log_mlflow(trained, params=None, run_id=None):
        from training.pipeline import evaluate, log_run

        metrics = evaluate(trained["workdir"])
        log.info("metrics: %s", metrics)
        mlflow_run_id = log_run(
            trained["workdir"],
            trained["data_path"],
            PROJECT_ROOT,
            EXPERIMENT,
            params["model_type"],
            params["seed"],
            run_name=f"airflow-{params['model_type']}-{run_id}",
            extra_tags={"airflow.dag_run_id": run_id, "airflow.dag_id": "training_pipeline"},
        )
        log.info("logged MLflow run %s", mlflow_run_id)
        return {"metrics": metrics, "mlflow_run_id": mlflow_run_id}

    @task
    def quality_gate(evaluated, params=None):
        from training.pipeline import QualityGateError
        from training.pipeline import quality_gate as run_gate

        try:
            result = run_gate(evaluated["metrics"], params["min_f1"])
        except QualityGateError as exc:
            log.error("%s", exc)
            raise AirflowFailException(str(exc)) from None
        log.info("quality gate passed: %s", result)
        return evaluated["mlflow_run_id"]

    @task
    def register_or_promote(mlflow_run_id):
        from training.pipeline import register_model

        result = register_model(mlflow_run_id)
        log.info("registered: %s", result)
        return result

    validated = validate_data()
    prepared = preprocess(validated)
    trained = train_model(prepared)
    evaluated = evaluate_log_mlflow(trained)
    gate = quality_gate(evaluated)
    register_or_promote(gate)


training_pipeline()
