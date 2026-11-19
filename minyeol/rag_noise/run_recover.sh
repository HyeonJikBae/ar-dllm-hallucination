#!/bin/bash
# v7 이 끝나기를 기다렸다가 Qwen 오류 복구 (PREREG_poison.md v9), GPU 0.
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs; P=logs/PROGRESS_recover.txt
until grep -q "ALL DONE" logs/PROGRESS_knock.txt 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') start" >> $P
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/minyeol/.conda/envs/sae-dream/bin/python -u patch_recover.py > logs/recover_qwen.log 2>&1
echo "$(date "+%F %T") end (exit $?)" >> $P; echo "$(date '+%F %T') ALL DONE" >> $P
