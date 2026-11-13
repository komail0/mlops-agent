import ast
from pathlib import Path

import pytest

from training.make_data import build
from training.pipeline import (
    QualityGateError,
    ValidationError,
    evaluate,
    preprocess,
    quality_gate,
    train_model,
    validate_dataset,
)

DAG_FILE = Path(__file__).resolve().parents[1] / "dags" / "training_pipeline.py"


@pytest.fixture
def good_csv(tmp_path):
    path = tmp_path / "good.csv"
    build("v1").to_csv(path, index=False)
    return path


def test_validate_accepts_good_data(good_csv):
    report = validate_dataset(good_csv)
    assert report["rows"] == 569 and report["min_class_share"] > 0.3


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda df: df.drop(columns="target"), "no 'target'"),
        (lambda df: df.head(10), "rows"),
        (lambda df: df.assign(**{"mean radius": None}), "missing values"),
        (lambda df: df.assign(target=2), "only 0 and 1"),
        (lambda df: df.assign(target=[1] * (len(df) - 5) + [0] * 5), "smallest class"),
    ],
)
def test_validate_rejects_bad_data(tmp_path, mutate, message):
    path = tmp_path / "bad.csv"
    mutate(build("v1")).to_csv(path, index=False)
    with pytest.raises(ValidationError, match=message):
        validate_dataset(path)


def test_validate_rejects_missing_and_unreadable(tmp_path):
    with pytest.raises(ValidationError, match="not found"):
        validate_dataset(tmp_path / "nope.csv")
    junk = tmp_path / "junk.csv"
    junk.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValidationError):
        validate_dataset(junk)


def test_stages_run_end_to_end(good_csv, tmp_path):
    work = tmp_path / "work"
    info = preprocess(good_csv, work)
    assert info["train_rows"] + info["test_rows"] == 569
    train_model(work, "logreg")
    metrics = evaluate(work)
    assert metrics["f1"] > 0.9 and metrics["n_test"] == info["test_rows"]
    assert (work / "metrics.json").is_file()


def test_degraded_data_fails_the_gate(tmp_path):
    path = tmp_path / "v2.csv"
    build("v2").to_csv(path, index=False)
    work = tmp_path / "work"
    preprocess(path, work)
    train_model(work)
    metrics = evaluate(work)
    with pytest.raises(QualityGateError, match="quality gate failed"):
        quality_gate(metrics, 0.8)


def test_quality_gate_passes_and_handles_missing_metric():
    assert quality_gate({"f1": 0.9}, 0.8)["passed"] is True
    with pytest.raises(QualityGateError):
        quality_gate({}, 0.8)


# ---- DAG structure, checked statically so Airflow is not needed in CI -----------------


@pytest.fixture(scope="module")
def dag_tree():
    return ast.parse(DAG_FILE.read_text(encoding="utf-8"))


def _dag_body(tree):
    func = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "training_pipeline"
    )
    return func


def test_dag_declares_retries_and_id(dag_tree):
    source = DAG_FILE.read_text(encoding="utf-8")
    assert 'dag_id="training_pipeline"' in source
    default_args = next(
        n.value
        for n in ast.walk(dag_tree)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "default_args"
    )
    values = {
        ast.literal_eval(k): v for k, v in zip(default_args.keys, default_args.values, strict=True)
    }
    assert ast.literal_eval(values["retries"]) >= 1
    assert "retry_delay" in values


def test_dag_task_order_matches_required_stages(dag_tree):
    body = _dag_body(dag_tree)
    flow = [
        (stmt.value.func.id, [a.id for a in stmt.value.args if isinstance(a, ast.Name)])
        for stmt in body.body
        if isinstance(stmt, ast.Assign | ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
    ]
    order = [name for name, _ in flow]
    assert order == [
        "validate_data",
        "preprocess",
        "train_model",
        "evaluate_log_mlflow",
        "quality_gate",
        "register_or_promote",
    ]
    # each stage consumes the previous stage's output (dependency chain)
    for (_, args), previous in zip(flow[1:], [n for n, _ in flow][:-1], strict=True):
        assert args, f"{previous} output is not consumed"


def test_dag_failures_do_not_retry_and_gate_blocks_promotion(dag_tree):
    source = DAG_FILE.read_text(encoding="utf-8")
    assert source.count("raise AirflowFailException") >= 2  # validation and gate
    body = _dag_body(dag_tree)
    last = body.body[-1]
    assert isinstance(last, ast.Expr) and last.value.func.id == "register_or_promote"
    assert [a.id for a in last.value.args] == ["gate"]


def test_pipeline_module_has_no_airflow_import():
    text = (DAG_FILE.parents[1] / "training" / "pipeline.py").read_text(encoding="utf-8")
    assert "airflow" not in text
