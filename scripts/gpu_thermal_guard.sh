#!/usr/bin/env bash
# Pause GPU jobs (SIGSTOP) when the GPU gets too hot; resume (SIGCONT) once it cools.
# Usage: bash scripts/gpu_thermal_guard.sh [pause_at=85] [resume_at=75]
HOT=${1:-85}; COOL=${2:-75}; PAT='compare_models|train_xray_multitask|evaluate_ordinal_test|train_fusion'
paused=0
while true; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  if [ "$paused" = 0 ] && [ "${t:-0}" -ge "$HOT" ]; then
    pgrep -f "$PAT" | xargs -r kill -STOP; paused=1
    echo "$(date +%H:%M:%S) ${t}C >= ${HOT}C: paused training"
  elif [ "$paused" = 1 ] && [ "${t:-99}" -le "$COOL" ]; then
    pgrep -f "$PAT" | xargs -r kill -CONT; paused=0
    echo "$(date +%H:%M:%S) ${t}C <= ${COOL}C: resumed training"
  elif [ "$paused" = 1 ]; then
    pgrep -f "$PAT" | xargs -r kill -STOP   # catch any newly spawned worker
  fi
  sleep 10
done
