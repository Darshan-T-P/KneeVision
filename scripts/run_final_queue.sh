#!/usr/bin/env bash
cd "$(dirname "$0")/.."
uv run python scripts/compare_models.py --models densenet121 --mixup-alpha 0 --seed 7 --tag focal_nomixup_seed7
python3 - <<'P'
import re
p="scripts/run_seed_test_evals.sh"; s=open(p).read()
s=s.replace('"best_densenet121_focal_nomixup_seed123"','"best_densenet121_focal_nomixup_seed7"')
open("/tmp/claude-1000/seed7_eval.sh","w").write(s.replace('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"','SCRIPT_DIR="'+__import__("os").getcwd()+'/scripts"'))
P
bash /tmp/claude-1000/seed7_eval.sh > logs/seed7_focal_eval.log 2>&1
uv run python scripts/evaluate_xai_faithfulness.py --n 100 --scorecam-n 20 --lime-n 30
echo FINAL_QUEUE_DONE
