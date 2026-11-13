#!/usr/bin/env bash
# Build a kubeconfig for the agent's restricted ServiceAccount (short-lived token).
# The output file holds a credential: keep it out of Git (kubeconfig* is gitignored).
#   bash scripts/agent_kubeconfig.sh [namespace] [output-file]
set -euo pipefail
NS="${1:-mlops-agent}"
OUT="${2:-kubeconfig-agent}"
SERVER="$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')"
CA="$(kubectl config view --minify --raw -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')"
TOKEN="$(kubectl -n "$NS" create token mlops-agent-sa --duration=24h)"
umask 077
cat > "$OUT" <<YAML
apiVersion: v1
kind: Config
clusters:
  - name: agent-cluster
    cluster:
      server: ${SERVER}
      certificate-authority-data: ${CA}
users:
  - name: mlops-agent-sa
    user:
      token: ${TOKEN}
contexts:
  - name: agent
    context:
      cluster: agent-cluster
      user: mlops-agent-sa
      namespace: ${NS}
current-context: agent
YAML
echo "wrote $OUT (token valid 24h). Use it with: KUBECONFIG=$OUT"
