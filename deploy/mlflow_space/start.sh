#!/usr/bin/env bash
# Serve a read-mostly snapshot of the KneeVision MLflow store.
# The DB is copied to /tmp on every start, so anything a visitor creates or deletes
# in the UI disappears on the next restart; the committed snapshot is never modified.
set -euo pipefail
cp mlflow.db /tmp/mlflow.db
exec mlflow server \
  --backend-store-uri sqlite:////tmp/mlflow.db \
  --no-serve-artifacts \
  --host 0.0.0.0 --port 7860 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*" \
  --x-frame-options NONE
