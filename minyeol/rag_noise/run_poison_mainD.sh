#!/bin/bash
# poison 본 실행 D 부분 (619 질의 x gold / R / R_sw / D / D_sw): Qwen -> Dream (steps 16), 배치 1, GPU 0.  채점은 A 설계를 고정한 뒤 (PREREG_poison.md).
# 시작: setsid nohup bash run_poison_mainD.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_poison_mainD.txt
for m in qwen dream; do
  echo "$(date '+%F %T') start $m" >> $P
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u gen.py --model $m --data poison_mainD --tag poisonD --bs 1 > logs/gen_${m}_poisonD.log 2>&1
  rc=$?; echo "$(date "+%F %T") end $m (exit $rc)" >> $P
done
echo "$(date '+%F %T') ALL DONE" >> $P
