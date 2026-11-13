"""Train a classifier on the DVC-tracked dataset and log results with provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.provenance import dataset_provenance

LABEL = "target"


def build_model(model_type: str, seed: int):
    if model_type == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    raise ValueError(f"unknown model type '{model_type}'")


def train(data_path: str | Path, model_type: str = "logreg", seed: int = 42):
    """Return (fitted model, metrics dict). Metrics are computed on a held-out split."""
    frame = pd.read_csv(data_path)
    if LABEL not in frame.columns:
        raise ValueError(f"dataset has no '{LABEL}' column")
    x, y = frame.drop(columns=[LABEL]), frame[LABEL]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = build_model(model_type, seed)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "f1": float(f1_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    return model, metrics


def log_to_mlflow(
    metrics: dict,
    params: dict,
    tags: dict,
    experiment: str,
    model=None,
    run_name: str | None = None,
) -> str:
    """Log one run to MLflow and return its run id. MLflow is imported lazily."""
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        if model is not None:
            import mlflow.sklearn

            mlflow.sklearn.log_model(model, name="model")
        return run.info.run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-out", default=None, help="write metrics as JSON")
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    parser.add_argument("--experiment", default="mlops-agent-demo")
    parser.add_argument("--repo", default=".", help="Git/DVC project root for provenance")
    args = parser.parse_args(argv)

    model, metrics = train(args.data, args.model, args.seed)
    print(json.dumps(metrics))
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.mlflow:
        tags = dataset_provenance(args.data, args.repo)
        params = {"model_type": args.model, "seed": args.seed}
        run_id = log_to_mlflow(
            metrics, params, tags, args.experiment, model, os.environ.get("RUN_NAME")
        )
        print(f"mlflow run: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
