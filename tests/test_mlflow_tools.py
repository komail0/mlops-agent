import pytest

pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from agent.config import Config  # noqa: E402
from agent.schemas import ToolError  # noqa: E402
from agent.tools.mlflow_tools import MlflowTools  # noqa: E402

DAY_MS = 86_400_000
NOW = 1_800_000_000_000


@pytest.fixture
def env(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    cfg = Config(mlflow_tracking_uri=uri, min_f1=0.8, model_max_age_days=30)
    tools = MlflowTools(cfg, client=client, now_ms=lambda: NOW)
    return client, tools


def add_run(client, exp_id, name, f1, version, age_days=1, extra=None, params=None):
    run = client.create_run(
        exp_id,
        start_time=NOW - int(age_days * DAY_MS),
        tags={"mlflow.runName": name, "dataset.version": version, "dvc.md5": "m-" + version},
    )
    rid = run.info.run_id
    if f1 is not None:
        client.log_metric(rid, "f1", f1)
    client.log_metric(rid, "n_test", 143)
    for k, v in (params or {"model_type": "logreg"}).items():
        client.log_param(rid, k, v)
    for k, v in (extra or {}).items():
        client.log_metric(rid, k, v)
    client.set_terminated(rid, "FINISHED")
    return rid


@pytest.fixture
def seeded(env):
    client, tools = env
    exp = client.create_experiment("demo")
    ids = {
        "a": add_run(client, exp, "a", 0.95, "v1", age_days=5),
        "b": add_run(client, exp, "b", 0.70, "v2", age_days=3, params={"model_type": "rf"}),
        "c": add_run(client, exp, "c", 0.95, "v1", age_days=2),
        "nometric": add_run(client, exp, "n", None, "v1", age_days=1),
    }
    return client, tools, exp, ids


def test_list_experiments(seeded):
    _, tools, exp, _ = seeded
    names = [e["name"] for e in tools.list_experiments()["experiments"]]
    assert "demo" in names


def test_list_runs_newest_first_with_dataset(seeded):
    _, tools, _, ids = seeded
    runs = tools.list_runs("demo", limit=10)["runs"]
    assert runs[0]["run_id"] == ids["nometric"]
    first_a = next(r for r in runs if r["run_id"] == ids["a"])
    assert first_a["dataset"]["dataset.version"] == "v1"
    assert first_a["metrics"]["f1"] == 0.95
    assert len(tools.list_runs("demo", limit=2)["runs"]) == 2


def test_get_run_has_metrics_params_artifacts(seeded):
    _, tools, _, ids = seeded
    info = tools.get_run(ids["b"])
    assert info["params"]["model_type"] == "rf"
    assert info["metrics"]["f1"] == 0.70
    assert "artifacts" in info
    assert all(not k.startswith("mlflow.") for k in info["tags"])


def test_best_run_max_with_ties_and_caveats(seeded):
    _, tools, _, ids = seeded
    out = tools.best_run("demo", "f1", "max")
    assert out["best_run"]["run_id"] in (ids["a"], ids["c"])
    caveats = out["caveats"]
    assert len(caveats["tied_with_best"]) == 1
    assert caveats["runs_missing_metric"] == 1
    assert caveats["candidates_use_different_datasets"] is True
    assert caveats["passes_quality_gate"] is True
    assert "production-ready" in out["note"]


def test_best_run_min_mode(seeded):
    _, tools, _, ids = seeded
    assert tools.best_run("demo", "f1", "min")["best_run"]["run_id"] == ids["b"]


def test_best_run_flags_failed_gate_and_stale_model(env):
    client, tools = env
    exp = client.create_experiment("old")
    add_run(client, exp, "x", 0.6, "v2", age_days=90)
    caveats = tools.best_run("old", "f1")["caveats"]
    assert caveats["passes_quality_gate"] is False
    assert caveats["older_than_freshness_policy"] is True


def test_best_run_errors(seeded):
    _, tools, _, _ = seeded
    with pytest.raises(ToolError, match="no finished runs"):
        tools.best_run("demo", "does_not_exist")
    with pytest.raises(ToolError, match="not found"):
        tools.best_run("missing-experiment", "f1")
    with pytest.raises(ToolError, match="mode"):
        tools.best_run("demo", "f1", "median")


def test_compare_runs(seeded):
    _, tools, _, ids = seeded
    out = tools.compare_runs([ids["a"], ids["b"]])
    assert out["metrics"]["f1"] == {ids["a"]: 0.95, ids["b"]: 0.70}
    assert "model_type" in out["params_that_differ"]
    assert out["datasets"] == {ids["a"]: "v1", ids["b"]: "v2"}
    with pytest.raises(ToolError):
        tools.compare_runs([ids["a"]])
    with pytest.raises(ToolError, match="not found"):
        tools.compare_runs([ids["a"], "0" * 32])


def test_compare_experiments(seeded):
    client, tools, _, _ = seeded
    other = client.create_experiment("other")
    add_run(client, other, "z", 0.5, "v2")
    out = tools.compare_experiments(["demo", "other"], "f1")
    assert out["experiments"]["demo"]["best_value"] == 0.95
    assert out["experiments"]["other"]["best_value"] == 0.5
    assert out["experiments"]["demo"]["runs_with_metric"] == 3


def test_registry_info_empty_and_populated(seeded):
    client, tools, _, ids = seeded
    assert tools.registry_info()["available"] is False
    client.create_registered_model("clf")
    client.create_model_version("clf", source="file:///tmp/none", run_id=ids["a"])
    out = tools.registry_info("clf")
    assert out["available"] is True
    assert out["models"][0]["versions"][0]["run_id"] == ids["a"]
    assert tools.registry_info("nope")["available"] is False


def test_server_down_becomes_tool_error():
    cfg = Config(mlflow_tracking_uri="http://127.0.0.1:9", tool_timeout_s=2)
    tools = MlflowTools(cfg)
    with pytest.raises(ToolError, match="unavailable"):
        tools.list_experiments()


def test_specs(env):
    _, tools = env
    specs = tools.specs()
    assert len(specs) >= 3
    assert all(not s.mutating for s in specs)
    assert {s.name for s in specs} >= {"mlflow_best_run", "mlflow_compare_runs"}
