#!/usr/bin/env bash
# Overnight queue: extra CORAL seeds, second Focal seed, fusion retrain on the CORAL encoder.
# Each step is independent (no set -e) so one failure does not block the rest.
cd "$(dirname "$0")/.."
LOCKED=models/best_densenet121_ordinal_ordinal_soft_mixup_a04.pt
for seed in 7 2024; do
  uv run python scripts/compare_models.py --models densenet121 --ordinal --soft-ordinal-targets \
    --mixup-alpha 0.4 --seed $seed --tag ordinal_soft_mixup_a04_seed$seed
done
uv run python scripts/compare_models.py --models densenet121 --mixup-alpha 0 --seed 123 --tag focal_nomixup_seed123
uv run python scripts/train_fusion.py --image-model $LOCKED --out models/best_fusion_coral.pt --epochs 10 --seed 42
uv run python scripts/evaluate_fusion.py --fusion-model models/best_fusion_coral.pt --image-model $LOCKED \
  --out-dir reports/evaluate_fusion_coral --no-mlflow
echo QUEUE_DONE
