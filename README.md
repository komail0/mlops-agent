# MLOps Agent

A conversational agent that answers questions about an ML system by calling real tools for
DVC (dataset versions), MLflow (experiments), Apache Airflow (training pipeline) and
Kubernetes (deployed inference service).

Work in progress. Setup and usage sections are added as each module lands.

## Development

```
pip install -e ".[dev]"
ruff check .
pytest
```
