#!/usr/bin/env bash
# Extra seeds of the multitask + soft CORAL + MixUp recipe, each test-evaluated once.
cd "$(dirname "$0")/.."
for seed in 7 2024; do
  tag=ordinal_soft_mixup_a04_seed$seed
  N=best_densenet121_multitask_$tag
  uv run python scripts/train_xray_multitask.py --ordinal --soft-ordinal-targets --mixup-alpha 0.4 \
    --warmup-cosine --epochs 100 --seed $seed --tag $tag
  uv run python scripts/evaluate_ordinal_test.py --checkpoint models/$N.pt --tag $N \
    --out-json reports/test_evals/$N.test.json --out-md reports/test_evals/$N.test.md --save-predictions
done
echo MT_SEEDS_DONE
