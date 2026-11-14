#!/usr/bin/env bash
# Deliberately inject (and undo) the failures used to test the agent's diagnosis.
# Run from the repository root with the local stack up. Needs kubectl, docker and DVC.
#
#   bash scripts/failures.sh k8s-bad-config      F1: ConfigMap points to a missing model file
#   bash scripts/failures.sh k8s-restore
#   bash scripts/failures.sh data-corrupt        F2: dataset has missing values (breaks validation)
#   bash scripts/failures.sh data-restore
#   bash scripts/failures.sh mlflow-stop         F3: MLflow server is down
#   bash scripts/failures.sh mlflow-start
#
# Each failure is reversible; the *-restore / *-start commands return to the healthy state.
set -euo pipefail

NS="${K8S_NAMESPACE:-mlops-agent}"
PYTHON="${PYTHON:-python}"
DATA="${DATA_FILE:-data/dataset.csv}"

wait_pods() {
  kubectl -n "$NS" get pods -l app.kubernetes.io/name=inference-api
}

case "${1:-}" in
  k8s-bad-config)
    kubectl -n "$NS" patch configmap inference-api-config --type merge \
      -p '{"data":{"MODEL_PATH":"/app/models/missing-model.joblib"}}'
    # Recreate the pods so they start with the broken configuration.
    kubectl -n "$NS" delete pod -l app.kubernetes.io/name=inference-api --wait=false
    sleep 25
    wait_pods
    ;;
  k8s-restore)
    kubectl -n "$NS" patch configmap inference-api-config --type merge \
      -p '{"data":{"MODEL_PATH":"/app/models/model.joblib"}}'
    kubectl -n "$NS" delete pod -l app.kubernetes.io/name=inference-api --wait=false
    kubectl -n "$NS" rollout status deployment/inference-api --timeout=120s
    wait_pods
    ;;
  data-corrupt)
    "$PYTHON" - "$DATA" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
frame = pd.read_csv(path)
frame.loc[frame.index[:25], frame.columns[0]] = None  # blank a block of values
frame.to_csv(path, index=False)
print(f"blanked 25 values in {path}")
PY
    ;;
  data-restore)
    "$PYTHON" -m dvc checkout --force "$DATA"
    ;;
  mlflow-stop)
    docker compose -f infra/docker-compose.yml stop mlflow
    ;;
  mlflow-start)
    docker compose -f infra/docker-compose.yml start mlflow
    ;;
  *)
    sed -n 2,12p "$0"
    exit 2
    ;;
esac
