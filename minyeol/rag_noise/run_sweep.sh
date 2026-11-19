#!/bin/bash
# 실험 A (병렬도 sweep) + C (궤적): Dream, 기준 세트, 생성 16 토큰, steps 16 / 8 / 4 / 2 (단계당 1 / 2 / 4 / 8 토큰), --trace.
# GPU 0 에 Dream 두 프로세스: [steps 16] 과 [8 -> 4 -> 2].  시작: setsid nohup bash run_sweep.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_sweep.txt
run() { echo "$(date '+%F %T') start steps $1" >> $P
        CUDA_VISIBLE_DEVICES=0 $PY -u gen.py --model dream --data anchor --tag anchor --steps $1 --trace > logs/gen_dream_anchor_s$1.log 2>&1
        rc=$?; echo "$(date "+%F %T") end steps $1 (exit $rc)" >> $P; }
run 16 & 
( run 8; run 4; run 2 ) &
wait
echo "$(date '+%F %T') ALL DONE" >> $P
