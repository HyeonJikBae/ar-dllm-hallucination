#!/bin/bash
# 기준 세트 (1,000 질의 x closed/gold/hard3/rand3): Qwen -> Dream, GPU 0.  시작: setsid nohup bash run_anchor.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_anchor.txt
for m in qwen dream; do
  echo "$(date '+%F %T') start $m" >> $P
  CUDA_VISIBLE_DEVICES=0 $PY -u gen.py --model $m --data anchor --tag anchor > logs/gen_${m}_anchor.log 2>&1
  rc=$?; echo "$(date "+%F %T") end $m (exit $rc)" >> $P
done
echo "$(date '+%F %T') ALL DONE" >> $P
