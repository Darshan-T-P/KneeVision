#!/usr/bin/env bash
# Backbone comparison under the locked recipe (soft CORAL, MixUp 0.4, seed 42).
# Each model is trained once; a failure (e.g. GPU OOM) retries at batch 16, then moves on.
# Each resulting checkpoint is test-evaluated exactly once.
cd "$(dirname "$0")/.."
TAG=ordinal_soft_mixup_a04
cooldown() {  # break between models: wait (max 15 min) for the GPU to drop to <= 65C
  for _ in $(seq 90); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    [ "${t:-0}" -le 65 ] && break
    sleep 10
  done
  echo "[cooldown] GPU at ${t}C"
}
for model in convnext_tiny efficientnet-b4 swin_t vit_b_16; do
  N=best_${model}_ordinal_${TAG}
  if [ -f reports/test_evals/$N.test.json ]; then echo "[skip] $model already evaluated"; continue; fi
  cooldown
  run() { uv run python scripts/compare_models.py --models $model --ordinal --soft-ordinal-targets \
            --mixup-alpha 0.4 --seed 42 --tag $TAG --resume "$@"; }
  run || { echo "[retry] $model at batch 16"; run --batch-size 16; }
  if [ -f models/$N.pt ]; then
    uv run python scripts/evaluate_ordinal_test.py --checkpoint models/$N.pt --tag $N \
      --out-json reports/test_evals/$N.test.json --out-md reports/test_evals/$N.test.md --save-predictions
  else
    echo "[skip eval] $N.pt not produced"
  fi
  echo "[done] $model"
done
echo BACKBONES_DONE
