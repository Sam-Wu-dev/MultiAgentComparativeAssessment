#!/bin/bash
# run_all_topicalchat.sh
set -euo pipefail

DATA_ROOT="/train-data1/shaosen/USR-TopicalChat/dataFolder/topicalchat"
OUT_ROOT="/home/shaosen/LLM/Multi-agent/Saving/TopicalChat/all"
# BUDGET=48   # or adjust per experiment

for ctx_dir in "$DATA_ROOT"/*; do
  if [ -d "$ctx_dir" ]; then
    ctx_id=$(basename "$ctx_dir")
    echo "[info] Processing TopicalChat context: $ctx_id"

    for metric_dir in "$ctx_dir"/*; do
      if [ -d "$metric_dir" ]; then
        metric=$(basename "$metric_dir")
        echo "  [metric] $metric"
        python main_topicalChat.py \
          --metric_dir "$metric_dir" \
          --round_root "$OUT_ROOT/$ctx_id" \
          # --budget $BUDGET
      fi
    done
  fi
done
