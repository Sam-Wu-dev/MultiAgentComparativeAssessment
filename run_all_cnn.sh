#!/bin/bash
# run_all_cnn.sh
set -euo pipefail

DATA_ROOT="/train-data1/shaosen/SummEvalDataSet/SummEval/dataFolder/cnn"
OUT_ROOT="/home/shaosen/LLM/Multi-agent/Saving/SummEval/cnn_CH"
BUDGET=48   # adjust as needed

AGG_OUT_ROOT="$OUT_ROOT"
AGG_JSON="$OUT_ROOT/spearman_summary.json"

mkdir -p "$OUT_ROOT"

for article_dir in "$DATA_ROOT"/*; do
  if [ -d "$article_dir" ]; then
    article_id=$(basename "$article_dir")
    echo "[info] Processing CNN article: $article_id"

    mkdir -p "$OUT_ROOT/$article_id"

    for metric_dir in "$article_dir"/*; do
      if [ -d "$metric_dir" ]; then
        metric=$(basename "$metric_dir")
        echo "  [metric] $metric"

        python main_summeval.py \
          --metric_dir "$metric_dir" \
          --round_root "$OUT_ROOT/$article_id" \
          --budget "$BUDGET"

        # Aggregate immediately after each metric completes
        echo "  [aggregate] Updating Spearman Aggregation JSON..."
        python aggregate_summeval_spearman_json.py \
          --out_root "$AGG_OUT_ROOT" \
          --out_json "$AGG_JSON"
      fi
    done
  fi
done

echo "[done] All articles processed and aggregated at: $AGG_JSON"
