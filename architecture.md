# Architecture

## Overview

```
 user ──► CLI (agent/main.py) ──► Orchestrator ──► LLM (OpenAI-compatible, tool calling)
                                      │
                                      ▼
                       validate → safety checks → run tool (time limit)
                                      │
        ┌───────────────┬─────────────┼───────────────┬────────────────┐
        ▼               ▼             ▼               ▼
      DVC tools     MLflow tools  Airflow tools   Kubernetes tools
      (6)           (7)           (6)             (11)
        │               │             │               │
   Git + .dvc      MLflow server   Airflow 3      kind cluster
   (data-v1/v2)    (Docker)        REST API       (RBAC account)
                                   (Docker)
```

The agent loop is written directly on the OpenAI-compatible SDK, with no agent framework. The
model chooses tools; the code decides whether a call may run.

## Orchestrator ([agent/orchestrator.py](agent/orchestrator.py))

One user message runs a loop: ask the model, run the tool calls it returns, feed the results
back, repeat until it answers. Limits keep a turn bounded: at most `MAX_STEPS` model rounds and
`RUN_TIMEOUT_S` seconds, after which the model is forced to answer from what it has. Every tool
call is checked, timed (default `TOOL_TIMEOUT_S`, per-tool override), logged and recorded with
its arguments and output as evidence.

## Tool contract

Every tool is a `ToolSpec` (name, description, JSON-schema parameters, handler, `mutating`,
timeout). Handlers return a `ToolResult {ok, tool, data, error, duration_ms}`. Failures are
values, not exceptions, and a failed result tells the model the information is unavailable and
must not be invented. Arguments are validated against the schema before the handler runs.

| System | Read tools | Changing tools |
|---|---|---|
| DVC | list datasets, dataset version, compare versions, check changes, reproduce instructions | `dvc_checkout_version` |
| MLflow | list experiments, list runs, get run, best run, compare runs, compare experiments, registry info | none |
| Airflow | list DAGs, DAG runs, run detail, failed tasks, task logs | `airflow_trigger_dag` |
| Kubernetes | deployments, pods, pod status, pod logs, services, events, ConfigMaps, unhealthy pods, deployed version | `k8s_restart_deployment`, `k8s_scale_deployment` |

## Safety (enforced in code, not only in the prompt)

- **Allowlists.** Kubernetes namespaces (`K8S_ALLOWED_NAMESPACES`), Airflow DAGs
  (`ALLOWED_DAGS`) and DVC arguments are checked before any call. Scale is limited to 1 to 5
  replicas. Trigger accepts only the keys `data_path`, `model_type`, `seed`, `min_f1`.
- **Two-step confirmation.** A changing tool first returns `confirmation_required` and a token.
  Only text the user types as `confirm <token>` approves it, and the orchestrator handles that
  path itself, so the model cannot approve its own action. A token is single use, tied to the
  tool and its exact arguments, and expires after `CONFIRM_TTL_S`.
- **Redaction.** Secrets and credential-shaped values are masked in logs, tool output and
  ConfigMap data before the model sees them.
- **Least privilege.** The agent uses the ServiceAccount `mlops-agent-sa`. Its Role can read
  pods, logs, events, services and ConfigMaps, and patch deployments and their scale, only in
  the `mlops-agent` namespace. It cannot read Secrets. Even if code checks failed, the cluster
  would refuse everything else.

## Evidence rules ([agent/prompts.py](agent/prompts.py))

The system prompt requires every operational claim to come from a tool result, requires a
minimum tool coverage for broad questions (retraining needs MLflow, Airflow and Kubernetes;
dataset questions need the DVC change check), forbids starting a training run on its own, and
fixes the answer format: Known facts, Uncertainties, Recommended actions. Tools back this up
with caveats in their data: tied or near-tied scores, runs on different dataset versions, run
age against the freshness policy, and the F1 quality gate (`MIN_F1`).

## Data and model lineage

```
data/dataset.csv ──(DVC, md5)──► data/dataset.csv.dvc ──(Git tag data-v1 / data-v2)
        │
        ▼
 training/train.py ──► MLflow run: metrics + tags
        dataset.version, dvc.md5, dvc.dataset_commit, git.commit,
        dvc.uncommitted_change, dvc.workspace_matches
        │
        ▼
 Airflow DAG training_pipeline ──► registry ──► models/model.joblib ──► image ──► Kubernetes
```

The dataset version is derived by matching the file's md5 to the `data-v*` tags, so a run
records the data it really used, including when the workspace file was edited after the last
commit.

## Training pipeline ([dags/training_pipeline.py](dags/training_pipeline.py))

`validate_data → preprocess → train_model → evaluate_log_mlflow → quality_gate →
register_or_promote`. Validation and the quality gate raise `AirflowFailException`, so a bad
dataset or a weak model stops the run and the later tasks show `upstream_failed`. Other tasks
retry twice.

## Deployment ([kubernetes/](kubernetes/), [api/main.py](api/main.py))

A FastAPI inference service (2 replicas, readiness and liveness probes, resource limits,
settings from a ConfigMap, ClusterIP service) on a kind cluster. The model file path comes from
the ConfigMap, which is how a wrong value makes pods fail readiness, a failure the agent can
trace with the ConfigMap tool.

## Observability

Logs are JSON lines in `logs/agent.jsonl` with a request id per user message: model rounds,
each tool call (arguments, status, duration) and confirmation events, all redacted.

## Known limits

- The DVC remote is a local folder, so data is rebuilt, not pulled, on a new machine.
- The agent's Kubernetes token lasts 24 hours and must be regenerated afterwards.
- DVC commands are slow on this machine (12 to 26 seconds), so DVC tools have a 90 second limit.
- Model quality depends on the LLM; the code limits what a wrong decision can do, not what the
  model says.
