#!/usr/bin/env bash
cd "$(dirname "$0")/.."
D=sae_analysis/results_twelve/dream_ext
CUDA_VISIBLE_DEVICES=0 STEP0_ROWS=$D/rows.jsonl STEP0_ANSWERS=$D/topk_margin_s32.jsonl STEP0_OUT=sae_analysis/features_dream_step0_ext /home/minyeol/.conda/envs/sae-dream/bin/python sae_analysis/dream_step0_extract.py
echo "$(date '+%F %T') extract end"; echo ALL_DONE
