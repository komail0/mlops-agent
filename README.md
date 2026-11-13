# MLOps Agent

A conversational agent that answers questions about an ML system by calling real tools for
DVC (dataset versions), MLflow (experiments), Apache Airflow (training pipeline) and
Kubernetes (deployed inference service).

Work in progress. Setup and usage sections are added as each module lands.

## Local services (MLflow and Airflow 3)

Needs Docker. Secrets live in `infra/.env`, which is not tracked.

```
cp infra/.env.example infra/.env     # then replace every value (see comments in the file)
cd infra
docker compose up -d --build
```

| Service | URL |
|---|---|
| MLflow tracking server | http://localhost:5000 |
| Airflow 3 (UI and REST API `/api/v2`) | http://localhost:8080 |

Airflow uses the built-in simple auth manager. Request a token with
`POST /auth/token` (`{"username": ..., "password": ...}`) and send it as
`Authorization: Bearer <token>`. Unauthenticated calls return 401.

## Development

```
pip install -e ".[dev]"
ruff check .
pytest
```
