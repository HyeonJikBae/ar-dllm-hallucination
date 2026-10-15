#!/usr/bin/env bash
# Dream topk_margin 확장 생성 -> 같은 신호 추출. VS Code를 꺼도 계속 돈다.
#   setsid nohup bash sae_analysis/run_dream_ext.sh > sae_analysis/logs/run_dream_ext.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PY=/home/minyeol/.conda/envs/sae-dream/bin/python
D=sae_analysis/results_twelve/dream_ext
echo "$(date '+%F %T') generate start"
CUDA_VISIBLE_DEVICES=0 $PY sae_analysis/dream_topk_extend.py --n-facts 4000
echo "$(date '+%F %T') generate end ($(wc -l < $D/topk_margin_s32.jsonl) rows)"
CUDA_VISIBLE_DEVICES=0 STEP0_ROWS=$D/rows.jsonl STEP0_ANSWERS=$D/topk_margin_s32.jsonl STEP0_OUT=sae_analysis/features_dream_step0_ext $PY sae_analysis/dream_step0_extract.py
echo "$(date '+%F %T') extract end"
echo ALL_DONE
