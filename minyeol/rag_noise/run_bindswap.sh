#!/bin/bash
# 연결 교환 실험 (data/musique_bindswap{_pilot}.jsonl x S / U / U_sw / L / L_sw): Qwen -> Dream (steps 16), 배치 1, GPU 0.
# 사용: setsid nohup bash run_bindswap.sh bindswap_pilot > /dev/null 2>&1 < /dev/null &   (본 실행은 인자 bindswap)
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; D=$1; P=logs/PROGRESS_$D.txt
for m in qwen dream; do
  echo "$(date '+%F %T') start $m" >> $P
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u gen.py --model $m --data $D --tag $D --bs 1 > logs/gen_${m}_$D.log 2>&1
  rc=$?; echo "$(date "+%F %T") end $m (exit $rc)" >> $P
done
echo "$(date '+%F %T') ALL DONE" >> $P
