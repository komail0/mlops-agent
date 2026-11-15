"""Probes that go straight to the orchestrator (no language model involved)."""

import json
import os

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
marker = "/tmp/probe_marker"

CALLS = [
    (
        "B8 revision with injected command",
        "dvc_compare_versions",
        {"rev_a": f"main; touch {marker}", "rev_b": "data-v2"},
    ),
    (
        "B8 revision with path traversal",
        "dvc_compare_versions",
        {"rev_a": "../../etc", "rev_b": "data-v2"},
    ),
    ("B8 path outside the project", "dvc_dataset_version", {"path": "../secret.csv"}),
    (
        "B7 run id that does not exist",
        "mlflow_get_run",
        {"run_id": "0123456789abcdef0123456789abcdef"},
    ),
    (
        "C best run by a metric no run has",
        "mlflow_best_run",
        {"experiment": "mlops-agent-demo", "metric": "val_loss", "mode": "min", "limit": 5},
    ),
    (
        "C last five runs, only three asked-for exist",
        "mlflow_list_runs",
        {"experiment": "mlops-agent-demo", "limit": 5},
    ),
]
for label, name, args in CALLS:
    r = agent._dispatch(ToolCall(id="x", name=name, arguments=args))
    body = json.dumps(r.data)[:260] if r.ok else r.error
    print(f"## {label}\n   {name} {json.dumps(args)}\n   ok={r.ok} -> {body}\n")
print("injected command ran:", os.path.exists(marker))
