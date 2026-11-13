"""Assemble the tool registry from the four tool modules."""

from __future__ import annotations

from agent.config import Config
from agent.safety import ConfirmationManager
from agent.schemas import ToolSpec
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DvcTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MlflowTools


def build_registry(config: Config, confirmations: ConfirmationManager) -> dict[str, ToolSpec]:
    specs: list[ToolSpec] = []
    specs += DvcTools(config, confirmations).specs()
    specs += MlflowTools(config).specs()
    specs += AirflowTools(config, confirmations).specs()
    specs += KubernetesTools(config, confirmations).specs()
    registry = {s.name: s for s in specs}
    if len(registry) != len(specs):
        raise ValueError("duplicate tool names")
    return registry
