# MLOps Agent

A conversational agent that answers questions about an ML system by calling real tools for
**DVC** (dataset versions), **MLflow** (experiments and registry), **Apache Airflow 3**
(training pipeline) and **Kubernetes** (the deployed inference service). It answers only from
tool results, says so when a system is unavailable, and changes nothing without an explicit
user confirmation. See [architecture.md](architecture.md) for the design.

## Requirements

- Python 3.11+ (3.14 works, see the DVC note below), Git
- Docker Desktop (about 8 GB RAM for the stack), `kubectl`, [kind](https://kind.sigs.k8s.io/)
- An OpenAI-compatible LLM endpoint with tool calling (the project was developed against
  Gemini's OpenAI-compatible API)

## Install

```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,train,data,api]"
```

On Python 3.14 the `dvc` build can fail inside pip's isolated environment. If it does:
`pip install --no-build-isolation "dvc>=3"`.

## Configure

Secrets are never committed. Both files below are gitignored.

1. `infra/.env` for the Docker services: `cp infra/.env.example infra/.env`, then replace every
   value (the file explains how to generate each one).
2. `.env` in the repository root for the agent:

```
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=...
LLM_MODEL=gemini-3.1-flash-lite
AIRFLOW_USERNAME=...        # same values as infra/.env
AIRFLOW_PASSWORD=...
```

Optional settings (defaults in [agent/config.py](agent/config.py)): `MLFLOW_TRACKING_URI`,
`AIRFLOW_BASE_URL`, `K8S_ALLOWED_NAMESPACES`, `ALLOWED_DAGS`, `MAX_STEPS`, `TOOL_TIMEOUT_S`,
`RUN_TIMEOUT_S`, `CONFIRM_TTL_S`, `MODEL_MAX_AGE_DAYS`, `MIN_F1`.

## Start the system

**1. Dataset versions (DVC).** The two dataset versions are tagged `data-v1` and `data-v2`. The
DVC remote is a local folder that is not in Git, so rebuild the data on a new machine. The
generator is deterministic and the script checks each result against the hash recorded at the tag:

```
bash scripts/setup_data.sh        # PYTHON=.venv/bin/python if the venv is not active
```

It leaves the workspace on the current version (`data-v2`) with both versions in the DVC cache, so
the agent can check out v1 later. `python -m training.seed_mlflow` (step 2) regenerates both
versions itself, and records the dataset tag and hash for each run.

**2. MLflow and Airflow.**

```
cd infra && docker compose up -d --build && cd ..
python -m training.seed_mlflow                # real runs on v1 and v2
```

| Service | URL |
|---|---|
| MLflow | http://localhost:5000 |
| Airflow 3 UI and REST API (`/api/v2`) | http://localhost:8080 |

**3. Kubernetes (kind).**

```
kind create cluster --config kubernetes/kind-config.yaml
python -m training.export_model --version v2 --out models
docker build -t mlops-agent-api:1.0.0 .
kind load docker-image mlops-agent-api:1.0.0 --name mlops-agent
kubectl apply -f kubernetes/namespace.yaml -f kubernetes/configmap.yaml \
  -f kubernetes/rbac.yaml -f kubernetes/service.yaml -f kubernetes/deployment.yaml
bash scripts/agent_kubeconfig.sh              # writes kubeconfig-agent (24 h token)
export KUBECONFIG="$PWD/kubeconfig-agent"     # the agent uses its restricted account
```

The agent connects as `mlops-agent-sa`, which can read pods, logs, events, services, deployments
and ConfigMaps in the `mlops-agent` namespace and patch deployments only. Secrets are not
readable.

## Use the agent

```
python -m agent.main                          # interactive chat
python -m agent.main --once "Is the deployment healthy?"
python -m agent.main --script demo.txt --evidence
```

`--evidence` prints every tool call with its arguments and redacted output. A script holds one
user message per line; the line `@confirm` types `confirm <token>` for the token the previous
answer asked for, and `@wait N` pauses. Structured JSON logs go to `logs/agent.jsonl`.

Example questions:

- What is the best experiment and which dataset version did it use?
- Which dataset version is deployed, and is the deployment healthy?
- Does the model need retraining? (checks MLflow, Airflow and Kubernetes)
- Why did the last pipeline run fail?
- Scale the inference deployment to 3 replicas. (asks for `confirm <token>` first)

## Failure injection

`scripts/failures.sh` breaks and repairs the system on purpose, to test the agent's diagnosis:
`k8s-bad-config` / `k8s-restore`, `data-corrupt` / `data-restore`, `mlflow-stop` /
`mlflow-start`.

## Report and evidence

- [23I-0717-A-REPORT.pdf](23I-0717-A-REPORT.pdf): the technical report (architecture, tools and
  permissions, evidence from all four systems, five tested conversations, two failure
  investigations, limitations).
- [screenshots/](screenshots/): MLflow and Airflow UI screenshots, DVC and Kubernetes captures.
- [evidence/](evidence/): recorded conversations with every tool call and its output
  (`python -m agent.main --script FILE --evidence`), and the out-of-policy calls the code rejects
  (`evidence/scenarios/s6b_direct_calls.py`).

## Development

```
ruff check .
pytest
```

CI runs both on every push and pull request ([.github/workflows/ci.yml](.github/workflows/ci.yml)).
The tests use fakes for the external systems, so they need no running services.
