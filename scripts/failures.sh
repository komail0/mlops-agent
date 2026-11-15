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
#   bash scripts/failures.sh k8s-bad-probe       readiness probe points to a path that returns 404
#   bash scripts/failures.sh k8s-bad-image       image tag that does not exist (ImagePullBackOff)
#   bash scripts/failures.sh k8s-oom             memory limit too small (OOMKilled)
#   bash scripts/failures.sh k8s-heal            re-apply kubernetes/deployment.yaml
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
  k8s-bad-probe)
    kubectl -n "$NS" patch deployment inference-api --type json -p       '[{"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe/httpGet/path","value":"/does-not-exist"}]'
    sleep 30
    wait_pods
    ;;
  k8s-bad-image)
    kubectl -n "$NS" set image deployment/inference-api api=mlops-agent-api:9.9.9
    sleep 30
    wait_pods
    ;;
  k8s-oom)
    kubectl -n "$NS" patch deployment inference-api --type json -p       '[{"op":"replace","path":"/spec/template/spec/containers/0/resources/requests/memory","value":"8Mi"},{"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/memory","value":"24Mi"}]'
    sleep 40
    wait_pods
    ;;
  k8s-heal)
    kubectl apply -f kubernetes/deployment.yaml
    kubectl -n "$NS" rollout status deployment/inference-api --timeout=180s
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
    sed -n 2,16p "$0"
    exit 2
    ;;
esac
