import io
import json
import re
import time
from unittest.mock import MagicMock

import pytest

from agent.config import Config
from agent.llm import LLMClient, LLMError, LLMReply, ToolCall
from agent.logging_setup import setup_logging
from agent.orchestrator import Orchestrator
from agent.safety import ConfirmationManager
from agent.schemas import ToolError, ToolSpec, object_schema
from agent.tools.kubernetes_tools import KubernetesTools

SECRET = "sk-" + "z9y8x7w6" * 4


class ScriptedLLM:
    """A fake model: returns scripted replies, or computes one from the conversation."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools=None):
        self.seen.append((list(messages), tools))
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply


def call(name, args=None, id_="c1"):
    return ToolCall(id=id_, name=name, arguments=args or {})


def tools_reply(*calls):
    return LLMReply(content="", tool_calls=list(calls))


def text(content):
    return LLMReply(content=content)


def spec(name, handler, params=None, mutating=False):
    return ToolSpec(name, name, params or object_schema(), handler, mutating=mutating)


def make(llm, specs, **cfg):
    config = Config(tool_timeout_s=cfg.pop("tool_timeout_s", 2), **cfg)
    registry = {s.name: s for s in specs}
    cm = cfg.pop("cm", None) or ConfirmationManager()
    return Orchestrator(config, llm, registry, cm), cm


def tool_messages(llm_call_index, llm):
    msgs = llm.seen[llm_call_index][0]
    return [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]


# ---- tool selection and execution -------------------------------------------------


def test_model_selected_tool_runs_and_result_reaches_the_answer():
    llm = ScriptedLLM(
        tools_reply(call("dvc_list_datasets")),
        lambda msgs: text("Found " + json.loads(msgs[-1]["content"])["data"]["name"]),
    )
    orch, _ = make(
        llm,
        [
            spec("dvc_list_datasets", lambda: {"name": "dataset-x"}),
            spec("k8s_list_pods", lambda: {}),
        ],
    )
    result = orch.chat("which datasets?")
    assert result.status == "ok" and result.answer == "Found dataset-x"
    assert [(t.name, t.ok) for t in result.tools] == [("dvc_list_datasets", True)]
    assert {t["function"]["name"] for t in llm.seen[0][1]} == {"dvc_list_datasets", "k8s_list_pods"}


def test_multi_tool_request_uses_earlier_results_to_choose_arguments():
    seen_args = {}

    def get_run(run_id):
        seen_args["run_id"] = run_id
        return {"dataset": "v1"}

    llm = ScriptedLLM(
        tools_reply(call("mlflow_best_run")),
        lambda msgs: tools_reply(
            call(
                "mlflow_get_run",
                {"run_id": json.loads(msgs[-1]["content"])["data"]["run_id"]},
                "c2",
            )
        ),
        tools_reply(
            call("k8s_list_deployments", id_="c3"), call("airflow_dag_runs", {"dag_id": "d"}, "c4")
        ),
        text("**Known facts**\n- done"),
    )
    orch, _ = make(
        llm,
        [
            spec("mlflow_best_run", lambda: {"run_id": "run-42"}),
            spec(
                "mlflow_get_run", get_run, object_schema({"run_id": {"type": "string"}}, ["run_id"])
            ),
            spec("k8s_list_deployments", lambda: {"ready": 2}),
            spec(
                "airflow_dag_runs",
                lambda dag_id: {"state": "success"},
                object_schema({"dag_id": {"type": "string"}}, ["dag_id"]),
            ),
        ],
    )
    result = orch.chat("find best model, dataset, deployment and pipeline status")
    assert seen_args["run_id"] == "run-42"  # taken from the first tool's result
    assert [t.name for t in result.tools] == [
        "mlflow_best_run",
        "mlflow_get_run",
        "k8s_list_deployments",
        "airflow_dag_runs",
    ]
    assert result.steps == 4


# ---- invalid requests, timeouts, failures ----------------------------------------


@pytest.mark.parametrize(
    "bad_call, fragment",
    [
        (call("no_such_tool"), "unknown tool"),
        (
            ToolCall("c1", "t", {}, parse_error="tool arguments were not valid JSON"),
            "not valid JSON",
        ),
        (call("t", {}), "missing required"),
        (call("t", {"x": "a", "extra": 1}), "unknown argument"),
        (call("t", {"x": 5}), "must be string"),
    ],
)
def test_invalid_requests_become_errors_and_the_loop_continues(bad_call, fragment):
    handler = MagicMock(return_value={})
    llm = ScriptedLLM(tools_reply(bad_call), text("could not run it"))
    orch, _ = make(llm, [spec("t", handler, object_schema({"x": {"type": "string"}}, ["x"]))])
    result = orch.chat("do it")
    payload = tool_messages(1, llm)[0]
    assert payload["ok"] is False and fragment in payload["error"]
    assert result.status == "ok" and result.answer == "could not run it"
    handler.assert_not_called()


def test_tool_timeout_is_reported_not_raised():
    llm = ScriptedLLM(tools_reply(call("slow")), text("it timed out"))
    orch, _ = make(llm, [spec("slow", lambda: time.sleep(2))], tool_timeout_s=0.1)
    started = time.monotonic()
    result = orch.chat("go")
    assert time.monotonic() - started < 1.5
    assert "timed out" in tool_messages(1, llm)[0]["error"]
    assert result.tools[0].ok is False


def test_service_failures_become_structured_errors_without_tracebacks():
    def down():
        raise ToolError("MLflow unavailable or request failed: ConnectionError")

    def crash():
        raise RuntimeError(f"secret detail {SECRET}")

    llm = ScriptedLLM(tools_reply(call("down"), call("crash", id_="c2")), text("two failures"))
    orch, _ = make(llm, [spec("down", down), spec("crash", crash)])
    result = orch.chat("check")
    payloads = tool_messages(1, llm)
    assert "MLflow unavailable" in payloads[0]["error"]
    assert payloads[1]["error"] == "internal error in tool (RuntimeError)"
    assert SECRET not in json.dumps(payloads)
    assert result.status == "ok"


# ---- no invented results ---------------------------------------------------------


def test_unavailable_tool_results_carry_no_values_and_an_explicit_instruction():
    """When every tool fails, the model receives only errors, and is told not to fill gaps."""

    def honest_model(msgs):
        results = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
        assert all(r["ok"] is False and r["data"] is None for r in results)
        assert all("Do not estimate" in r["instruction"] for r in results)
        return text("Unavailable: " + "; ".join(r["tool"] for r in results))

    fixture_value = "f1=0.9876-fixture"
    llm = ScriptedLLM(
        tools_reply(call("mlflow_best_run"), call("k8s_list_pods", id_="c2")), honest_model
    )
    orch, _ = make(
        llm,
        [
            spec("mlflow_best_run", lambda: (_ for _ in ()).throw(ToolError("MLflow unavailable"))),
            spec(
                "k8s_list_pods",
                lambda: (_ for _ in ()).throw(ToolError("Kubernetes API unavailable")),
            ),
        ],
    )
    result = orch.chat("best model and pods?")
    assert fixture_value not in json.dumps(llm.seen[1][0])
    assert result.answer.startswith("Unavailable")
    assert [t.ok for t in result.tools] == [False, False]


def test_llm_outage_answer_states_unavailability_and_does_not_guess():
    class Down:
        def chat(self, *a, **k):
            raise LLMError("LLM request failed after 5 attempts")

    orch, _ = make(Down(), [spec("t", lambda: {})])
    result = orch.chat("status?")
    assert result.status == "llm_error"
    assert "could not reach the language model" in result.answer


# ---- limits ----------------------------------------------------------------------


def test_step_limit_forces_a_final_answer_without_tools():
    loops = [tools_reply(call("t", id_=f"c{i}")) for i in range(3)]
    llm = ScriptedLLM(*loops, text("partial answer from evidence"))
    orch, _ = make(llm, [spec("t", lambda: {"n": 1})], max_steps=3)
    result = orch.chat("loop forever")
    assert result.status == "step_limit" and result.answer == "partial answer from evidence"
    assert llm.seen[-1][1] is None  # the forced final turn offers no tools


def test_run_timeout_stops_the_loop():
    clock = iter([0, 0, 999, 999, 999, 999])
    llm = ScriptedLLM(tools_reply(call("t")), text("stopped"))
    config = Config(run_timeout_s=10, tool_timeout_s=1)
    orch = Orchestrator(
        config, llm, {"t": spec("t", lambda: {})}, ConfirmationManager(), clock=lambda: next(clock)
    )
    result = orch.chat("slow")
    assert result.status == "timeout" and result.answer == "stopped"


# ---- confirmation and unauthorised changes -----------------------------------------


def restart_tool(cm):
    performed = []

    def restart(deployment, confirmation_token=None):
        params = {"deployment": deployment}
        if not confirmation_token:
            token = cm.request("restart", params, "restart " + deployment)
            raise ToolError(f"confirmation_required: confirm token '{token}'")
        if not cm.consume(confirmation_token, "restart", params):
            raise ToolError("confirmation token is invalid, unapproved, expired or mismatched")
        performed.append(deployment)
        return {"restarted": deployment}

    schema = object_schema(
        {"deployment": {"type": "string"}, "confirmation_token": {"type": "string"}}, ["deployment"]
    )
    return spec("restart", restart, schema, mutating=True), performed


def token_from(llm, index):
    return json.loads(llm.seen[index][0][-1]["content"])["error"].split("'")[1]


def test_confirmation_flow_only_the_user_can_release_an_action():
    cm = ConfirmationManager()
    tool, performed = restart_tool(cm)
    llm = ScriptedLLM(
        tools_reply(call("restart", {"deployment": "api"})),
        lambda msgs: text(
            "please reply: confirm " + json.loads(msgs[-1]["content"])["error"].split("'")[1]
        ),
        # second user turn, after approval
        lambda msgs: tools_reply(
            call(
                "restart",
                {
                    "deployment": "api",
                    "confirmation_token": re.search(r"token (\w{8})", msgs[-1]["content"]).group(1),
                },
            )
        ),
        text("restarted"),
    )
    orch = Orchestrator(Config(tool_timeout_s=2), llm, {"restart": tool}, cm)
    first = orch.chat("restart api")
    assert performed == []  # nothing happened before the user confirmed
    token = first.answer.split()[-1]
    second = orch.chat(f"confirm {token}")
    assert performed == ["api"] and second.answer == "restarted"


def test_model_cannot_approve_its_own_request():
    cm = ConfirmationManager()
    tool, performed = restart_tool(cm)
    llm = ScriptedLLM(
        tools_reply(call("restart", {"deployment": "api"})),
        lambda msgs: tools_reply(
            call(
                "restart",
                {
                    "deployment": "api",
                    "confirmation_token": json.loads(msgs[-1]["content"])["error"].split("'")[1],
                },
                "c2",
            )
        ),
        text("tried"),
    )
    orch = Orchestrator(Config(tool_timeout_s=2), llm, {"restart": tool}, cm)
    orch.chat("restart api")
    assert performed == []
    assert "invalid" in tool_messages(2, llm)[1]["error"]


def test_unknown_or_expired_confirmation_never_reaches_the_model():
    llm = ScriptedLLM()
    orch, _ = make(llm, [])
    result = orch.chat("confirm deadbeef")
    assert result.status == "rejected" and "not valid" in result.answer
    assert llm.seen == []


def test_unauthorised_kubernetes_modification_is_rejected_end_to_end():
    cm = ConfirmationManager()
    apps, core = MagicMock(), MagicMock()
    k8s = KubernetesTools(Config(k8s_allowed_namespaces=("mlops-agent",)), cm, core, apps)
    specs = {s.name: s for s in k8s.specs()}
    llm = ScriptedLLM(
        tools_reply(
            call("k8s_restart_deployment", {"deployment": "coredns", "namespace": "kube-system"})
        ),
        tools_reply(
            call(
                "k8s_scale_deployment",
                {
                    "deployment": "api",
                    "replicas": 3,
                    "namespace": "default",
                    "confirmation_token": "deadbeef",
                },
                "c2",
            )
        ),
        text("refused"),
    )
    orch = Orchestrator(Config(tool_timeout_s=2), llm, specs, cm)
    result = orch.chat("restart coredns in kube-system, then scale api")
    errors = [json.loads(m["content"])["error"] for m in llm.seen[2][0] if m["role"] == "tool"]
    assert all("not approved" in e for e in errors) and len(errors) == 2
    apps.patch_namespaced_deployment.assert_not_called()
    apps.patch_namespaced_deployment_scale.assert_not_called()
    assert result.answer == "refused"


# ---- logging, history, llm wrapper ----------------------------------------------


def test_structured_logs_have_name_status_duration_request_id_and_no_secrets():
    buf = io.StringIO()
    setup_logging(stream=buf)
    llm = ScriptedLLM(tools_reply(call("ok_tool"), call("bad_tool", id_="c2")), text("done"))

    def bad():
        raise ToolError(f"auth failed token={SECRET}")

    orch, _ = make(
        llm, [spec("ok_tool", lambda: {"v": f"api_key={SECRET}"}), spec("bad_tool", bad)]
    )
    result = orch.chat("go")
    records = [json.loads(line) for line in buf.getvalue().splitlines()]
    tool_logs = [r for r in records if r["event"] == "tool_call"]
    assert {r["tool"]: r["status"] for r in tool_logs} == {"ok_tool": "ok", "bad_tool": "error"}
    assert all(r["request_id"] == result.request_id and "duration_ms" in r for r in tool_logs)
    assert SECRET not in buf.getvalue()
    assert {"request_start", "request_end"} <= {r["event"] for r in records}


def test_tool_output_is_redacted_before_the_model_sees_it():
    llm = ScriptedLLM(tools_reply(call("leaky")), text("ok"))
    orch, _ = make(
        llm, [spec("leaky", lambda: {"note": f"key {SECRET}", "password": "hunter2hunter2"})]
    )
    orch.chat("go")
    assert SECRET not in json.dumps(llm.seen[1][0]) and "hunter2" not in json.dumps(llm.seen[1][0])


def test_follow_up_turns_keep_context():
    llm = ScriptedLLM(text("first"), text("second"))
    orch, _ = make(llm, [])
    orch.chat("hello")
    orch.chat("and again")
    roles = [m["role"] for m in llm.seen[1][0]]
    assert roles == ["system", "user", "assistant", "user"]


def test_llm_client_retries_then_parses_tool_calls():
    import openai

    class Resp:
        def __init__(self, message):
            self.choices = [MagicMock(message=message, finish_reason="tool_calls")]

    message = MagicMock(content=None)
    message.tool_calls = [MagicMock(id="x", function=MagicMock(arguments='{"a": 1}'))]
    message.tool_calls[0].function.name = "t"
    message.tool_calls[0].extra_content = None
    request = MagicMock()
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        openai.APIConnectionError(request=request),
        Resp(message),
    ]
    sleeps = []
    llm = LLMClient(
        Config(llm_model="m"), client=client, sleep=sleeps.append, retry_delays=(0.5, 1)
    )
    reply = llm.chat([{"role": "user", "content": "x"}], [{"type": "function"}])
    assert sleeps == [0.5] and reply.tool_calls[0].arguments == {"a": 1}
    client.chat.completions.create.side_effect = openai.APIConnectionError(request=request)
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "x"}])


# ---- CLI helpers: evidence and scripted conversations ----------------------------


def test_tool_records_keep_redacted_arguments_and_output_as_evidence():
    llm = ScriptedLLM(tools_reply(call("t", {"x": f"key {SECRET}"})), text("ok"))
    schema = object_schema({"x": {"type": "string"}}, ["x"])
    orch, _ = make(llm, [spec("t", lambda x: {"value": 42}, schema)])
    record = orch.chat("go").tools[0]
    assert SECRET not in json.dumps(record.arguments) and '"value": 42' in record.output


def test_scripted_conversation_confirms_only_what_a_tool_requested(tmp_path, capsys):
    from agent.main import pending_token, run_script

    cm = ConfirmationManager()
    tool, performed = restart_tool(cm)
    llm = ScriptedLLM(
        tools_reply(call("restart", {"deployment": "api"})),
        text("reply with confirm"),
        lambda msgs: tools_reply(
            call(
                "restart",
                {
                    "deployment": "api",
                    "confirmation_token": re.search(r"token (\w{8})", msgs[-1]["content"]).group(1),
                },
            )
        ),
        text("restarted"),
    )
    orch = Orchestrator(Config(tool_timeout_s=2), llm, {"restart": tool}, cm)
    script = tmp_path / "chat.txt"
    script.write_text("# comment\nrestart api\n@confirm\n", encoding="utf-8")
    assert run_script(orch, script, evidence=True) == 0
    assert performed == ["api"]
    out = capsys.readouterr().out
    assert "you> restart api" in out and "you> confirm " in out and "--- evidence:" in out
    # @confirm with nothing pending is an error, not a blind approval
    bare = tmp_path / "bare.txt"
    bare.write_text("@confirm\n", encoding="utf-8")
    assert run_script(orch, bare, evidence=False) == 1
    assert pending_token(_empty_result()) is None


def _empty_result():
    from agent.orchestrator import AgentResult

    return AgentResult("x", "r", "ok")


def test_per_tool_timeout_overrides_the_default():
    slow = spec("slow", lambda: time.sleep(0.4) or {"done": True})
    slow.timeout_s = 3
    llm = ScriptedLLM(tools_reply(call("slow")), text("ok"))
    orch, _ = make(llm, [slow], tool_timeout_s=0.1)
    assert orch.chat("go").tools[0].ok is True
