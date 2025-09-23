#!/bin/bash
# run_all_cnn.sh
set -euo pipefail

DATA_ROOT="/train-data1/shaosen/SummEvalDataSet/SummEval/dataFolder/cnn"
OUT_ROOT="/home/shaosen/LLM/Multi-agent/Saving/SummEval/cnn_all"
BUDGET=96   # or adjust as you like

for article_dir in "$DATA_ROOT"/*; do
  if [ -d "$article_dir" ]; then
    article_id=$(basename "$article_dir")
    echo "[info] Processing CNN article: $article_id"

    for metric_dir in "$article_dir"/*; do
      if [ -d "$metric_dir" ]; then
        metric=$(basename "$metric_dir")
        echo "  [metric] $metric"
        python main_summeval.py \
          --metric_dir "$metric_dir" \
          --round_root "$OUT_ROOT/$article_id" \
          --budget $BUDGET
      fi
    done
  fi
done
