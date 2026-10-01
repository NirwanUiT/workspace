#!/usr/bin/env bash
# MMLU zero-shot calibration baseline on one free GPU, logged to the shared activity log.
# Run inside tmux:  tmux new -d -s llm-mmlu "bash scripts/run_mmlu_gpu.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
ACT=/mnt/nas1/nba055-2/idea_1/Nanograph/cluster/activity.log
export CUDA_VISIBLE_DEVICES=$(bash /mnt/nas1/nba055-2/idea_1/Nanograph/cluster/freegpu.sh 20) || exit 1
mkdir -p logs

run() {  # name model out
  local log=$PWD/logs/$1.log
  echo "$(date -Is) START claude-code llm-$1 host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES log=$log" >> "$ACT"
  .venv/bin/python scripts/mmlu_baseline.py --model "$2" --out "$3" > "$log" 2>&1
  local rc=$?
  echo "$(date -Is) END claude-code llm-$1 host=$(hostname) exit=$rc" >> "$ACT"
  return $rc
}

run mmlu-qwen3.5-4b      Qwen/Qwen3.5-4B      results/mmlu_zeroshot_qwen3.5-4b
run mmlu-qwen3.5-4b-base Qwen/Qwen3.5-4B-Base results/mmlu_zeroshot_qwen3.5-4b-base
