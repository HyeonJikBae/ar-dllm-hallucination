#!/bin/bash
# poison 파일럿 (50 질의 x gold / A / D): Qwen -> Dream (steps 16) -> 채점, GPU 0.  시작: setsid nohup bash run_poison_pilot.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_poison_pilot.txt
for m in qwen dream; do
  echo "$(date '+%F %T') start $m" >> $P
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u gen.py --model $m --data poison_pilot --tag poisonpilot > logs/gen_${m}_poisonpilot.log 2>&1
  rc=$?; echo "$(date "+%F %T") end $m (exit $rc)" >> $P
done
$PY grade_poison.py > logs/poison_pilot.txt 2>&1
echo "$(date '+%F %T') ALL DONE (grade exit $?)" >> $P
