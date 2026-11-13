import json

import httpx
import pytest

from agent.config import Config
from agent.safety import ConfirmationManager
from agent.schemas import ToolError
from agent.tools.airflow_tools import AirflowClient, AirflowTools

RUN_ID = "manual__2026-11-13T19:50:26.935432+00:00"
FAKE_TOKEN = "eyJ" + "a" * 20 + "." + "b" * 20 + "." + "c" * 20
PASSWORD = "pw-" + "z" * 12


class FakeAirflow:
    """Records requests and answers like the Airflow 3 REST API."""

    def __init__(self):
        self.requests = []
        self.token_calls = 0
        self.expire_next = False
        self.fail_with = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_with:
            raise self.fail_with
        path = request.url.path
        if path == "/auth/token":
            body = json.loads(request.content)
            if body["password"] != PASSWORD:
                return httpx.Response(401, json={"detail": "bad credentials"})
            self.token_calls += 1
            return httpx.Response(201, json={"access_token": FAKE_TOKEN})
        if self.expire_next:
            self.expire_next = False
            return httpx.Response(401, json={"detail": "expired"})
        if request.headers.get("authorization") != f"Bearer {FAKE_TOKEN}":
            return httpx.Response(401)
        if path == "/api/v2/dags":
            return httpx.Response(
                200,
                json={
                    "dags": [
                        {
                            "dag_id": "training_pipeline",
                            "is_paused": False,
                            "tags": [{"name": "ml"}],
                        },
                        {"dag_id": "other", "is_paused": True, "tags": []},
                    ]
                },
            )
        if path.endswith("/dagRuns") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "dag_runs": [
                        {"dag_run_id": RUN_ID, "state": "failed", "run_type": "manual", "conf": {}}
                    ]
                },
            )
        if path.endswith("/dagRuns") and request.method == "POST":
            body = json.loads(request.content)
            return httpx.Response(
                200, json={"dag_run_id": "manual__new", "state": "queued", "conf": body["conf"]}
            )
        if path.endswith("/taskInstances"):
            return httpx.Response(
                200,
                json={
                    "task_instances": [
                        {"task_id": "validate_data", "state": "success", "try_number": 1},
                        {
                            "task_id": "quality_gate",
                            "state": "failed",
                            "try_number": 1,
                            "max_tries": 2,
                        },
                        {
                            "task_id": "register_or_promote",
                            "state": "upstream_failed",
                            "try_number": 0,
                        },
                    ]
                },
            )
        if "/logs/" in path:
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"timestamp": "t1", "level": "info", "event": "quality gate evaluating"},
                        {
                            "timestamp": "t2",
                            "level": "error",
                            "event": f"failed token={FAKE_TOKEN}",
                            "error_detail": [
                                {
                                    "exc_type": "AirflowFailException",
                                    "exc_value": "quality gate failed: f1=0.69 is below 0.8",
                                    "frames": [{"filename": "x"}],
                                }
                            ],
                        },
                    ]
                },
            )
        if "/taskInstances/" in path:
            return httpx.Response(200, json={"task_id": "quality_gate", "try_number": 1})
        if "/dagRuns/" in path:
            return httpx.Response(200, json={"dag_run_id": RUN_ID, "state": "failed"})
        return httpx.Response(404)


def make(password=PASSWORD, ttl_clock=None):
    fake = FakeAirflow()
    cfg = Config(
        airflow_base_url="http://airflow.test",
        airflow_username="admin",
        airflow_password=password,
        allowed_dags=("training_pipeline",),
    )
    cm = ConfirmationManager()
    client = AirflowClient(cfg, transport=httpx.MockTransport(fake))
    return AirflowTools(cfg, cm, client), fake, cm


def posts(fake):
    return [r for r in fake.requests if r.method == "POST" and "dagRuns" in r.url.path]


def test_list_dags_marks_approved():
    tools, _, _ = make()
    dags = {d["dag_id"]: d for d in tools.list_dags()["dags"]}
    assert dags["training_pipeline"]["approved_for_trigger"] is True
    assert dags["other"]["approved_for_trigger"] is False


def test_token_is_fetched_once_and_sent_as_bearer():
    tools, fake, _ = make()
    tools.list_dags()
    tools.dag_runs("training_pipeline")
    assert fake.token_calls == 1
    assert all(
        r.headers["authorization"] == f"Bearer {FAKE_TOKEN}"
        for r in fake.requests
        if r.url.path != "/auth/token"
    )


def test_expired_token_is_refreshed_once():
    tools, fake, _ = make()
    tools.list_dags()
    fake.expire_next = True
    assert tools.list_dags()["count"] == 2
    assert fake.token_calls == 2


def test_authentication_failure_hides_password():
    tools, _, _ = make(password="wrong")
    with pytest.raises(ToolError, match="authentication failed") as err:
        tools.list_dags()
    assert "wrong" not in str(err.value)
    cfg = Config(airflow_base_url="http://airflow.test")
    with pytest.raises(ToolError, match="not configured"):
        AirflowTools(
            cfg,
            ConfirmationManager(),
            AirflowClient(cfg, transport=httpx.MockTransport(FakeAirflow())),
        ).list_dags()


def test_run_detail_and_failed_tasks_separate_root_cause():
    tools, _, _ = make()
    detail = tools.run_detail("training_pipeline", RUN_ID)
    assert [t["task_id"] for t in detail["tasks"]][:2] == ["validate_data", "quality_gate"]
    failed = tools.failed_tasks("training_pipeline")
    assert failed["dag_run_id"] == RUN_ID
    assert [t["task_id"] for t in failed["failed_tasks"]] == ["quality_gate"]
    assert [t["task_id"] for t in failed["blocked_downstream_tasks"]] == ["register_or_promote"]


def test_run_id_with_special_characters_is_quoted():
    tools, fake, _ = make()
    tools.run_detail("training_pipeline", RUN_ID)
    raw = next(r for r in fake.requests if "dagRuns/" in str(r.url)).url.raw_path.decode()
    assert "manual__2026-11-13T19%3A50%3A26.935432%2B00%3A00" in raw


def test_task_logs_extract_errors_and_redact():
    tools, _, _ = make()
    out = tools.task_logs("training_pipeline", RUN_ID, "quality_gate")
    assert out["try_number"] == 1
    assert out["errors"] == ["AirflowFailException: quality gate failed: f1=0.69 is below 0.8"]
    assert FAKE_TOKEN not in json.dumps(out)


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.dag_runs("../etc"),
        lambda t: t.run_detail("training_pipeline", "a b"),
        lambda t: t.task_logs("training_pipeline", RUN_ID, "x/../y"),
        lambda t: t.failed_tasks("dag;drop"),
    ],
)
def test_invalid_ids_rejected_before_any_request(call):
    tools, fake, _ = make()
    with pytest.raises(ToolError, match="invalid"):
        call(tools)
    assert fake.requests == []


def test_trigger_requires_confirmation_and_sends_nothing_first():
    tools, fake, cm = make()
    with pytest.raises(ToolError, match="confirmation_required") as err:
        tools.trigger_dag("training_pipeline")
    token = str(err.value).split("'")[1]
    assert posts(fake) == []
    with pytest.raises(ToolError, match="invalid"):
        tools.trigger_dag("training_pipeline", confirmation_token=token)  # not approved
    assert posts(fake) == []
    cm.approve(token)
    out = tools.trigger_dag("training_pipeline", confirmation_token=token)
    assert out["triggered"] is True and out["state"] == "queued"
    assert len(posts(fake)) == 1
    with pytest.raises(ToolError):  # single use
        tools.trigger_dag("training_pipeline", confirmation_token=token)


def test_trigger_rejects_unapproved_dag_and_forged_token():
    tools, fake, _ = make()
    with pytest.raises(ToolError, match="not approved"):
        tools.trigger_dag("other")
    with pytest.raises(ToolError, match="invalid"):
        tools.trigger_dag("training_pipeline", confirmation_token="deadbeef")
    assert posts(fake) == []


def test_trigger_token_is_bound_to_conf():
    tools, fake, cm = make()
    with pytest.raises(ToolError) as err:
        tools.trigger_dag("training_pipeline", {"model_type": "rf"})
    token = str(err.value).split("'")[1]
    cm.approve(token)
    with pytest.raises(ToolError, match="invalid"):
        tools.trigger_dag("training_pipeline", {"model_type": "logreg"}, token)
    assert posts(fake) == []


@pytest.mark.parametrize(
    "conf",
    [
        {"evil": 1},
        {"data_path": "../../etc/passwd"},
        {"data_path": "/abs/x.csv"},
        {"model_type": "gpt"},
    ],
)
def test_trigger_rejects_bad_conf(conf):
    tools, fake, _ = make()
    with pytest.raises(ToolError):
        tools.trigger_dag("training_pipeline", conf)
    assert posts(fake) == []


def test_service_failures_become_tool_errors():
    tools, fake, _ = make()
    fake.fail_with = httpx.ConnectError("refused")
    with pytest.raises(ToolError, match="unavailable"):
        tools.list_dags()
    fake.fail_with = httpx.ReadTimeout("slow")
    with pytest.raises(ToolError, match="timed out"):
        tools.dag_runs("training_pipeline")


def test_http_errors_are_reported():
    tools, _, _ = make()
    with pytest.raises(ToolError, match="not found"):
        tools.client.request("GET", "/api/v2/nothing")

    def boom(request):
        return (
            httpx.Response(500)
            if request.url.path != "/auth/token"
            else httpx.Response(201, json={"access_token": FAKE_TOKEN})
        )

    cfg = Config(airflow_base_url="http://airflow.test", airflow_username="a", airflow_password="b")
    broken = AirflowTools(
        cfg, ConfirmationManager(), AirflowClient(cfg, transport=httpx.MockTransport(boom))
    )
    with pytest.raises(ToolError, match="HTTP 500"):
        broken.list_dags()


def test_specs():
    tools, _, _ = make()
    specs = {s.name: s for s in tools.specs()}
    assert len(specs) >= 3
    assert specs["airflow_trigger_dag"].mutating is True
    assert sum(s.mutating for s in specs.values()) == 1
