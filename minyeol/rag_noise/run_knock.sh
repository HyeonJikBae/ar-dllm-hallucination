#!/bin/bash
# v6 이 끝나기를 기다렸다가 Dream attention knockout (PREREG_poison.md v7): none -> q -> ctrl, GPU 0.
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs; P=logs/PROGRESS_knock.txt
until grep -q "ALL DONE" logs/PROGRESS_samefact.txt 2>/dev/null; do sleep 60; done
for k in none q ctrl; do
  echo "$(date '+%F %T') start $k" >> $P
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/minyeol/.conda/envs/sae-dream/bin/python -u knock_bind.py --ko $k > logs/knock_$k.log 2>&1
  echo "$(date "+%F %T") end $k (exit $?)" >> $P
done
echo "$(date '+%F %T') ALL DONE" >> $P
