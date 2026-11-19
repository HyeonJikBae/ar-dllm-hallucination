#!/bin/bash
# padding 재검증: 기준 세트 gold / hard3 / rand3 를 배치 1 로 다시 생성 (Qwen -> Dream steps 16), GPU 0.
# 기존 실행은 배치 8 이라 Dream 답이 padding 길이에 따라 바뀔 수 있음 (2hop__689472_52835).  시작: setsid nohup bash run_anchor_bs1.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_anchor_bs1.txt
for m in qwen dream; do
  echo "$(date '+%F %T') start $m" >> $P
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u gen.py --model $m --data anchor --tag anchorbs1 --bs 1 --conds gold,hard3,rand3 > logs/gen_${m}_anchorbs1.log 2>&1
  rc=$?; echo "$(date "+%F %T") end $m (exit $rc)" >> $P
done
echo "$(date '+%F %T') ALL DONE" >> $P
