#!/usr/bin/env bash
# External evaluation on the independent Gornale & Patravali cohort (evaluation only; one pass per model/variant).
cd "$(dirname "$0")/.."
BASE="data/external/gornale/Digital Knee X-ray Images/Knee X-ray Images"
OUT=reports/external_gornale; mkdir -p $OUT
A=best_densenet121_ordinal_ordinal_soft_mixup_a04
M=best_densenet121_multitask_ordinal_soft_mixup_a04
run() { # name checkpoint expert variant [extra flags]
  local name=$1 ck=$2 ex=$3 var=$4; shift 4
  local flag=""; [ "$var" = pad ] && flag="--pad-square"
  uv run python scripts/evaluate_external.py --root "$BASE/expert_$ex/MedicalExpert-$ex" --checkpoint models/$ck.pt \
    $flag --save-predictions $OUT/${name}_expert${ex}_${var}.predictions.csv --out-json $OUT/${name}_expert${ex}_${var}.json "$@" > $OUT/${name}_expert${ex}_${var}.log 2>&1
  echo "[done] $name expert$ex $var"
}
for var in stretch pad; do
  for s in "" _seed7 _seed123 _seed2024; do run "plain${s:-_seed42}" ${A}${s} I $var; done
  for s in "" _seed7 _seed2024; do run "multitask${s:-_seed42}" ${M}${s} I $var; done
  run focal_nomixup_seed42 best_densenet121_focal_nomixup I $var --allow-non-ordinal
done
for var in stretch pad; do run plain_seed42 $A II $var; run multitask_seed42 $M II $var; done
echo EXTERNAL_DONE
