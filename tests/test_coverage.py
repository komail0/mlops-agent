from types import SimpleNamespace

from agent.config import Config
from agent.coverage import missing_coverage
from agent.llm import LLMReply, ToolCall
from agent.orchestrator import Orchestrator
from agent.safety import ConfirmationManager
from agent.schemas import ToolError, ToolSpec, object_schema

RETRAIN_Q = "Is retraining necessary for the best model?"


def rec(name, ok=True, output=""):
    return SimpleNamespace(name=name, ok=ok, output=output)


FAILED_RUNS = '{"ok": true, "data": {"runs": [{"dag_run_id": "r2", "state": "failed"}]}}'
GOOD_RUNS = '{"ok": true, "data": {"runs": [{"dag_run_id": "r2", "state": "success"}]}}'


# ---- the rule itself ---------------------------------------------------------------


def test_only_retraining_questions_need_coverage():
    assert missing_coverage("What is running on Kubernetes?", []) == []
    assert len(missing_coverage(RETRAIN_Q, [])) == 4


def test_each_missing_system_is_named():
    only_mlflow = missing_coverage(RETRAIN_Q, [rec("mlflow_best_run")])
    assert any("Airflow" in m for m in only_mlflow) and any("Kubernetes" in m for m in only_mlflow)
    assert not any("MLflow" in m for m in only_mlflow)


def test_failed_latest_run_requires_failed_tasks():
    base = [rec("mlflow_best_run"), rec("k8s_deployed_version"), rec("dvc_dataset_version")]
    failed = missing_coverage(RETRAIN_Q, [*base, rec("airflow_dag_runs", output=FAILED_RUNS)])
    assert failed == ["airflow_failed_tasks, because the latest pipeline run failed"]
    good = missing_coverage(RETRAIN_Q, [*base, rec("airflow_dag_runs", output=GOOD_RUNS)])
    assert good == []
    done = [*base, rec("airflow_dag_runs", output=FAILED_RUNS), rec("airflow_failed_tasks")]
    assert missing_coverage(RETRAIN_Q, done) == []


def test_a_failed_call_still_counts_as_checked():
    records = [
        rec("mlflow_list_experiments", ok=False),
        rec("airflow_dag_runs", ok=False),
        rec("k8s_list_deployments", ok=False),
        rec("dvc_check_changes", ok=False),
    ]
    assert missing_coverage(RETRAIN_Q, records) == []


# ---- the loop ---------------------------------------------------------------------


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def tools_reply(*names):
    return LLMReply(content="", tool_calls=[ToolCall(f"c{i}", n, {}) for i, n in enumerate(names)])


def text(content):
    return LLMReply(content=content)


def orchestrator(llm, failing=()):
    def make(name, data):
        def handler():
            if name in failing:
                raise ToolError("service unreachable")
            return data

        return ToolSpec(name, name, object_schema(), handler)

    specs = [
        make("mlflow_best_run", {"best": "run-1"}),
        make("airflow_dag_runs", {"runs": [{"dag_run_id": "r2", "state": "failed"}]}),
        make("airflow_failed_tasks", {"failed_tasks": ["quality_gate"]}),
        make("k8s_deployed_version", {"version": "1.0.0"}),
        make("dvc_dataset_version", {"version": "v1"}),
    ]
    return Orchestrator(
        Config(tool_timeout_s=2), llm, {s.name: s for s in specs}, ConfirmationManager()
    )


def test_early_answer_is_rejected_until_all_systems_are_checked():
    llm = ScriptedLLM(
        tools_reply("mlflow_best_run"),
        text("DRAFT one: no retraining needed."),
        tools_reply("airflow_dag_runs", "k8s_deployed_version", "dvc_dataset_version"),
        text("DRAFT two: still fine."),
        tools_reply("airflow_failed_tasks"),
        text("FINAL answer with all evidence."),
    )
    result = orchestrator(llm).chat(RETRAIN_Q)
    assert result.answer == "FINAL answer with all evidence."
    assert [t.name for t in result.tools] == [
        "mlflow_best_run",
        "airflow_dag_runs",
        "k8s_deployed_version",
        "dvc_dataset_version",
        "airflow_failed_tasks",
    ]
    # the model was told what was missing, and the rejected drafts are not in its history
    last = llm.seen[-1]
    assert not any("DRAFT" in str(m.get("content")) for m in last)
    nudges = [m["content"] for m in last if m["role"] == "user" and "still missing" in m["content"]]
    assert len(nudges) == 2 and "airflow_failed_tasks" in nudges[1]


def test_unreachable_systems_do_not_trap_the_loop():
    llm = ScriptedLLM(
        tools_reply(
            "mlflow_best_run", "airflow_dag_runs", "k8s_deployed_version", "dvc_dataset_version"
        ),
        text("MLflow, Airflow and Kubernetes are all unavailable."),
    )
    everything = (
        "mlflow_best_run",
        "airflow_dag_runs",
        "k8s_deployed_version",
        "dvc_dataset_version",
    )
    result = orchestrator(llm, failing=everything).chat(RETRAIN_Q)
    assert result.answer.startswith("MLflow, Airflow")
    assert len(llm.seen) == 2  # no nudge was needed


def test_other_questions_are_never_nudged():
    llm = ScriptedLLM(tools_reply("k8s_deployed_version"), text("Running 1.0.0."))
    result = orchestrator(llm).chat("What is running on Kubernetes?")
    assert result.answer == "Running 1.0.0."
    assert len(llm.seen) == 2


def test_nudges_are_capped_so_a_stubborn_model_still_gets_an_answer():
    llm = ScriptedLLM(text("a"), text("b"), text("c final"))
    result = orchestrator(llm).chat(RETRAIN_Q)
    assert result.status == "ok" and result.answer == "c final"
    assert result.tools == []  # the cap, not the tools, ended the loop
