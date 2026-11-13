"""Pipeline stages used by the Airflow DAG, written as plain functions so they can be tested
without Airflow: validate -> preprocess -> train -> evaluate -> quality gate -> register."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from training.provenance import dataset_provenance
from training.train import LABEL, build_model, log_to_mlflow

MODEL_NAME = "mlops-agent-classifier"


class ValidationError(ValueError):
    """The dataset is unfit for training. Retrying cannot fix it."""


class QualityGateError(RuntimeError):
    """The evaluated model did not meet the quality threshold. Promotion must not happen."""


def validate_dataset(path: str | Path, min_rows: int = 100, min_class_share: float = 0.2) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ValidationError(f"dataset not found: {p}")
    try:
        frame = pd.read_csv(p)
    except Exception as exc:
        raise ValidationError(f"dataset is not readable CSV: {type(exc).__name__}") from None
    if LABEL not in frame.columns:
        raise ValidationError(f"dataset has no '{LABEL}' column")
    if len(frame) < min_rows:
        raise ValidationError(f"dataset has {len(frame)} rows, need at least {min_rows}")
    if frame.isnull().any().any():
        raise ValidationError("dataset contains missing values")
    if set(frame[LABEL].unique()) - {0, 1}:
        raise ValidationError(f"'{LABEL}' must contain only 0 and 1")
    share = float(frame[LABEL].value_counts(normalize=True).min())
    if share < min_class_share:
        raise ValidationError(f"smallest class is {share:.0%} of rows, need {min_class_share:.0%}")
    return {"rows": len(frame), "columns": frame.shape[1], "min_class_share": round(share, 3)}


def preprocess(path: str | Path, workdir: str | Path, seed: int = 42) -> dict:
    frame = pd.read_csv(path)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(
        frame, test_size=0.25, random_state=seed, stratify=frame[LABEL]
    )
    train_df.to_csv(work / "train.csv", index=False)
    test_df.to_csv(work / "test.csv", index=False)
    return {"train_rows": len(train_df), "test_rows": len(test_df), "workdir": str(work)}


def train_model(workdir: str | Path, model_type: str = "logreg", seed: int = 42) -> str:
    work = Path(workdir)
    train_df = pd.read_csv(work / "train.csv")
    model = build_model(model_type, seed)
    model.fit(train_df.drop(columns=[LABEL]), train_df[LABEL])
    out = work / "model.joblib"
    joblib.dump(model, out)
    return str(out)


def evaluate(workdir: str | Path) -> dict:
    work = Path(workdir)
    model = joblib.load(work / "model.joblib")
    test_df = pd.read_csv(work / "test.csv")
    y = test_df[LABEL]
    pred = model.predict(test_df.drop(columns=[LABEL]))
    metrics = {
        "f1": float(f1_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "n_train": int(len(pd.read_csv(work / "train.csv"))),
        "n_test": int(len(test_df)),
    }
    (work / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return metrics


def log_run(
    workdir: str | Path,
    data_path: str | Path,
    repo: str | Path,
    experiment: str,
    model_type: str,
    seed: int,
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> str:
    """Log metrics, parameters, the model and dataset provenance to MLflow. Returns run id."""
    work = Path(workdir)
    metrics = json.loads((work / "metrics.json").read_text(encoding="utf-8"))
    model = joblib.load(work / "model.joblib")
    tags = {**dataset_provenance(data_path, repo), **(extra_tags or {})}
    return log_to_mlflow(
        metrics,
        {"model_type": model_type, "seed": seed},
        tags,
        experiment,
        model=model,
        run_name=run_name,
    )


def quality_gate(metrics: dict, min_f1: float) -> dict:
    f1 = metrics.get("f1")
    if f1 is None or f1 < min_f1:
        raise QualityGateError(f"quality gate failed: f1={f1} is below the minimum {min_f1}")
    return {"passed": True, "f1": f1, "min_f1": min_f1}


def register_model(run_id: str, alias: str = "candidate") -> dict:
    """Register the run's model and point an alias at the new version."""
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, alias, version.version)
    return {"model": MODEL_NAME, "version": version.version, "alias": alias}
