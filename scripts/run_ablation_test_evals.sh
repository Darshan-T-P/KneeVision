#!/usr/bin/env bash
# run_ablation_test_evals.sh
#
# Runs frozen held-out TEST evaluation for all 6 remaining ablation cells.
# Each checkpoint is evaluated exactly once. Results land in reports/test_evals/.
#
# Prerequisites: uv sync completed, .venv is healthy.
# Usage: bash scripts/run_ablation_test_evals.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${PROJECT_ROOT}/reports/test_evals"
mkdir -p "${OUT_DIR}"


# Focal/softmax checkpoints (non-ordinal) — need --allow-non-ordinal flag
NON_ORDINAL_CHECKPOINTS=(
    "best_densenet121"
    "best_densenet121_focal_nomixup"
    "best_densenet121_focal_mixup"
)

# CORAL ordinal checkpoints — evaluated without flag
ORDINAL_CHECKPOINTS=(
    "best_densenet121_ordinal_ordinal_soft_nomixup"
    "best_densenet121_ordinal_ordinal_soft_mixup"
    "best_densenet121_ordinal_ordinal_soft_mixup_a04_seed123"
)

run_eval() {
    local NAME="$1"
    local EXTRA_FLAGS="$2"
    local PT="${PROJECT_ROOT}/models/${NAME}.pt"

    if [[ ! -f "$PT" ]]; then
        echo "[SKIP] Checkpoint not found: $PT"
        return
    fi

    local OUT_JSON="${OUT_DIR}/${NAME}.test.json"
    local OUT_MD="${OUT_DIR}/${NAME}.test.md"

    if [[ -f "$OUT_JSON" ]]; then
        echo "[SKIP] Already evaluated: ${NAME} → $(basename "$OUT_JSON")"
        return
    fi

    echo ""
    echo "======================================================"
    echo " Evaluating: ${NAME}"
    echo "======================================================"
    uv run python "${SCRIPT_DIR}/evaluate_ordinal_test.py" \
        --checkpoint "$PT" \
        --tag "$NAME" \
        --out-json "$OUT_JSON" \
        --out-md  "$OUT_MD" \
        --save-plots \
        --plots-dir "${OUT_DIR}/plots" \
        --save-predictions \
        $EXTRA_FLAGS

    echo "[DONE] ${NAME}"
    echo "       JSON → ${OUT_JSON}"
    echo "       MD   → ${OUT_MD}"
}

for NAME in "${NON_ORDINAL_CHECKPOINTS[@]}"; do
    run_eval "$NAME" "--allow-non-ordinal"
done

for NAME in "${ORDINAL_CHECKPOINTS[@]}"; do
    run_eval "$NAME" ""
done

echo ""
echo "All evaluations complete. Results in: ${OUT_DIR}"
