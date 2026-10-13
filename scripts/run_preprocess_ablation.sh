#!/usr/bin/env bash
# Pre-specified preprocessing ablation (base-paper-style preprocessing), DenseNet121 + soft CORAL + MixUp 0.4, seed 42.
# Checkpoint choice uses OAI VALIDATION QWK only. Each model gets ONE OAI test pass and ONE external pass (stretch resize).
cd "$(dirname "$0")/.."
BASE="data/external/gornale/Digital Knee X-ray Images/Knee X-ray Images/expert_I/MedicalExpert-I"
OUT=reports/external_gornale_preprocess; mkdir -p $OUT
for v in hist_eq paper; do
  TAG=ordinal_soft_mixup_a04_pre_$v; N=best_densenet121_ordinal_$TAG
  [ -f reports/test_evals/$N.test.json ] || {
    uv run python scripts/compare_models.py --models densenet121 --ordinal --soft-ordinal-targets --mixup-alpha 0.4 \
      --seed 42 --preprocess $v --tag $TAG --resume
    uv run python scripts/evaluate_ordinal_test.py --checkpoint models/$N.pt --tag $N --preprocess $v \
      --out-json reports/test_evals/$N.test.json --out-md reports/test_evals/$N.test.md --save-predictions
  }
  uv run python scripts/evaluate_external.py --root "$BASE" --checkpoint models/$N.pt --preprocess $v \
    --save-predictions $OUT/${v}_seed42_expertI_stretch.predictions.csv --out-json $OUT/${v}_seed42_expertI_stretch.json > $OUT/${v}_seed42_expertI_stretch.log 2>&1
  echo "[done] $v"
done
echo PREPROCESS_DONE
