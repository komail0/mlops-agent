#!/usr/bin/env bash
# Rebuild both dataset versions on a new machine and put them in the local DVC cache.
# The DVC remote is a local folder that is not in Git, so the data is regenerated. The
# generator is deterministic, and the result is checked against the hash recorded at each tag.
# Ends on the current version (data-v2), so the working tree is clean.
#   bash scripts/setup_data.sh
set -euo pipefail
PYTHON="${PYTHON:-python}"
DVC_FILE="data/dataset.csv.dvc"

for version in v1 v2; do
  git checkout "data-$version" -- "$DVC_FILE"
  "$PYTHON" -m training.make_data --version "$version"
  "$PYTHON" -m dvc commit "$DVC_FILE"
  "$PYTHON" -m dvc status "$DVC_FILE"
done
git checkout HEAD -- "$DVC_FILE"
"$PYTHON" -m dvc status "$DVC_FILE"
