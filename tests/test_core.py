import io
import json

import pytest

from agent.config import load_config, load_dotenv
from agent.logging_setup import log_tool_call, new_request_id, setup_logging
from agent.redact import MASK, redact, redact_obj
from agent.safety import (
    ConfirmationManager,
    SafetyError,
    check_dag,
    check_namespace,
    validate_dvc_args,
)
from agent.schemas import ToolResult, ToolSpec, object_schema

# Fake secrets are assembled at runtime so no real-looking key is stored in the source.
FAKE_KEY = "sk-" + "a1b2c3d4" * 4
FAKE_GOOGLE = "AIza" + "x" * 30


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_tool_result_envelope():
    ok = ToolResult.success("t", {"a": 1})
    bad = ToolResult.failure("t", "boom")
    assert ok.to_dict()["ok"] is True and ok.to_dict()["data"] == {"a": 1}
    assert bad.to_dict()["ok"] is False and bad.to_dict()["error"] == "boom"
    assert set(ok.to_dict()) == {"ok", "tool", "data", "error", "duration_ms"}


def test_tool_spec_openai_format():
    spec = ToolSpec(
        "dvc_status",
        "Check data changes",
        object_schema({"path": {"type": "string"}}, ["path"]),
        handler=lambda path: path,
    )
    fn = spec.to_openai()["function"]
    assert fn["name"] == "dvc_status"
    assert fn["parameters"]["required"] == ["path"]
    assert fn["parameters"]["additionalProperties"] is False


def test_config_defaults_and_overrides():
    cfg = load_config({})
    assert cfg.k8s_allowed_namespaces == ("mlops-agent",)
    assert cfg.allowed_dags == ("training_pipeline",)
    cfg = load_config({"K8S_ALLOWED_NAMESPACES": "a, b", "MAX_STEPS": "5", "ALLOWED_DAGS": "x"})
    assert cfg.k8s_allowed_namespaces == ("a", "b")
    assert cfg.max_steps == 5 and cfg.allowed_dags == ("x",)


def test_config_repr_hides_secrets():
    cfg = load_config({"LLM_API_KEY": FAKE_KEY, "AIRFLOW_PASSWORD": "hunter2hunter2"})
    text = repr(cfg)
    assert FAKE_KEY not in text and "hunter2" not in text


def test_load_dotenv_does_not_override(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# c\nA=1\nB='two'\nC=keep\n", encoding="utf-8")
    env = {"C": "existing"}
    load_dotenv(f, env)
    assert env == {"C": "existing", "A": "1", "B": "two"}


def test_redact_text_secrets():
    text = f"key {FAKE_KEY} and {FAKE_GOOGLE} and Bearer abcdefghijklmnop"
    out = redact(text)
    assert FAKE_KEY not in out and FAKE_GOOGLE not in out and "abcdefghijklmnop" not in out
    assert MASK in out


def test_redact_key_value_and_objects():
    assert "p4ssw0rd" not in redact("password=p4ssw0rd host=db")
    assert "host=db" in redact("password=p4ssw0rd host=db")
    obj = {"api_key": "zzz", "nested": [{"token": "t0k"}, "ok"], "name": "model"}
    out = redact_obj(obj)
    assert out["api_key"] == MASK and out["nested"][0]["token"] == MASK
    assert out["name"] == "model" and out["nested"][1] == "ok"


def test_structured_log_has_required_fields_and_no_secrets():
    buf = io.StringIO()
    setup_logging(stream=buf)
    rid = new_request_id()
    log_tool_call(rid, "k8s_list_pods", "error", 42, error=f"auth failed token={FAKE_KEY}")
    record = json.loads(buf.getvalue().strip())
    assert record["tool"] == "k8s_list_pods"
    assert record["status"] == "error"
    assert record["duration_ms"] == 42
    assert record["request_id"] == rid
    assert FAKE_KEY not in buf.getvalue()


def test_namespace_and_dag_allowlists():
    assert check_namespace("mlops-agent", ["mlops-agent"]) == "mlops-agent"
    with pytest.raises(SafetyError):
        check_namespace("kube-system", ["mlops-agent"])
    assert check_dag("training_pipeline", ["training_pipeline"])
    with pytest.raises(SafetyError):
        check_dag("other_dag", ["training_pipeline"])


@pytest.mark.parametrize(
    "args",
    [
        ["push"],
        ["remote", "add", "x", "y"],
        ["list", ".", ";", "whoami"],
        ["status", "data; rm -rf /"],
        ["status", "$(whoami)"],
        ["list", "../secrets"],
        ["list", "/etc/passwd"],
        ["list", "C:/Windows"],
        ["diff", "a|b"],
        [],
    ],
)
def test_dvc_rejects_unsafe(args):
    with pytest.raises(SafetyError):
        validate_dvc_args(args)


def test_dvc_accepts_approved():
    assert validate_dvc_args(["diff", "v1", "v2", "--targets", "data/train.csv"])
    assert validate_dvc_args(["status"]) == ["status"]
    assert validate_dvc_args(["list", ".", "--dvc-only"])


def test_confirmation_flow_requires_approval():
    cm = ConfirmationManager()
    params = {"namespace": "mlops-agent", "deployment": "api"}
    token = cm.request("k8s_restart_deployment", params, "restart api")
    assert cm.consume(token, "k8s_restart_deployment", params) is False  # not approved yet
    assert cm.approve(token) is True
    assert cm.consume(token, "k8s_restart_deployment", params) is True
    assert cm.consume(token, "k8s_restart_deployment", params) is False  # single use


def test_confirmation_binds_action_and_params():
    cm = ConfirmationManager()
    params = {"namespace": "mlops-agent", "deployment": "api"}
    token = cm.request("k8s_restart_deployment", params, "restart")
    cm.approve(token)
    assert cm.consume(token, "k8s_scale_deployment", params) is False
    assert cm.consume(token, "k8s_restart_deployment", {**params, "deployment": "other"}) is False
    assert cm.consume(token, "k8s_restart_deployment", params) is True


def test_confirmation_unknown_and_expired():
    clock = Clock()
    cm = ConfirmationManager(ttl_s=10, clock=clock)
    assert cm.approve("nope") is False
    assert cm.consume("nope", "a", {}) is False
    token = cm.request("a", {}, "d")
    clock.now = 11
    assert cm.approve(token) is False
    assert cm.consume(token, "a", {}) is False
