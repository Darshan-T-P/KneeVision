#!/usr/bin/env bash
# Tier B (pre-specified): domain-robust training vs a matched short-schedule baseline.
# DenseNet121, soft CORAL + MixUp 0.4, --epochs 40 --patience 8, seeds 42 and 7. Checkpoints chosen on OAI val QWK.
# Each model: one OAI test pass here; external evaluation (zero-shot + adaptations) at the end.
cd "$(dirname "$0")/.."
for seed in 42 7; do
  for arm in base robust; do
    TAG=short40_${arm}_seed${seed}; N=best_densenet121_ordinal_$TAG
    FLAG=""; [ $arm = robust ] && FLAG="--robust-aug"
    [ -f reports/test_evals/$N.test.json ] && { echo "[skip] $N"; continue; }
    uv run python scripts/compare_models.py --models densenet121 --ordinal --soft-ordinal-targets --mixup-alpha 0.4 \
      --epochs 40 --patience 8 --seed $seed --tag $TAG $FLAG --resume
    uv run python scripts/evaluate_ordinal_test.py --checkpoint models/$N.pt --tag $N $FLAG \
      --out-json reports/test_evals/$N.test.json --out-md reports/test_evals/$N.test.md --save-predictions
    echo "[done] $TAG"
  done
done
uv run python scripts/evaluate_external_adapt.py --tier-b
echo TIERB_DONE
