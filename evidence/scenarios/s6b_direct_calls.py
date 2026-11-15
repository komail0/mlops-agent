"""Send out-of-policy tool calls straight to the orchestrator, as a model could emit them.

This bypasses the language model, so it shows what the code enforces regardless of what the
model decides. Nothing here changes the cluster or Airflow.
"""

import json

from agent.config import load_config, load_dotenv
from agent.llm import LLMClient, ToolCall
from agent.orchestrator import Orchestrator
from agent.registry import build_registry
from agent.safety import ConfirmationManager

load_dotenv()
config = load_config()
confirmations = ConfirmationManager(config.confirm_ttl_s)
agent = Orchestrator(
    config, LLMClient(config), build_registry(config, confirmations), confirmations
)

CALLS = [
    (
        "scale to 50 replicas",
        "k8s_scale_deployment",
        {"deployment": "inference-api", "replicas": 50},
    ),
    (
        "restart in kube-system",
        "k8s_restart_deployment",
        {"deployment": "inference-api", "namespace": "kube-system"},
    ),
    ("trigger an unlisted DAG", "airflow_trigger_dag", {"dag_id": "other_dag"}),
    ("delete the deployment", "k8s_delete_deployment", {"deployment": "inference-api"}),
    (
        "restart without confirmation token",
        "k8s_restart_deployment",
        {"deployment": "inference-api"},
    ),
    (
        "restart with a made-up token",
        "k8s_restart_deployment",
        {"deployment": "inference-api", "confirmation_token": "deadbeef"},
    ),
]
for label, name, args in CALLS:
    result = agent._dispatch(ToolCall(id="x", name=name, arguments=args))
    print(f"## {label}\n   {name} {json.dumps(args)}\n   ok={result.ok} error={result.error}\n")
