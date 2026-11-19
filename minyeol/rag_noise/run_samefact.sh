#!/bin/bash
# 같은 사실 donor 구간 patching (PREREG_poison.md v6): Qwen 360 쌍 -> Dream (Qwen 이 맞힌 쌍), GPU 0.
# 사용: setsid nohup bash run_samefact.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs; P=logs/PROGRESS_samefact.txt
for m in qwen dream; do
  echo "$(date '+%F %T') start $m" >> $P
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/minyeol/.conda/envs/sae-dream/bin/python -u patch_samefact.py --model $m > logs/samefact_$m.log 2>&1
  echo "$(date "+%F %T") end $m (exit $?)" >> $P
done
echo "$(date '+%F %T') ALL DONE" >> $P
