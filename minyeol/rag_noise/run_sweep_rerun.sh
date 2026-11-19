#!/bin/bash
# run_sweep.sh 에서 OOM 으로 빠진 steps 8 / 4 를 한 번에 하나씩 (Dream 1 프로세스, SAE 학습과 공유).  끝나면 분석 다시.
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_sweep.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 8 4; do
  echo "$(date '+%F %T') start steps $s (rerun)" >> $P
  CUDA_VISIBLE_DEVICES=0 $PY -u gen.py --model dream --data anchor --tag anchor --steps $s --trace > logs/gen_dream_anchor_s$s.log 2>&1; rc=$?
  echo "$(date '+%F %T') end steps $s (exit $rc)" >> $P
done
{ $PY grade_anchor.py --sweep; echo; echo "===== 궤적 (H3) ====="; $PY trace_anchor.py; } > logs/analysis.txt 2>&1
echo "ANALYSIS DONE $(date '+%F %T')" >> logs/analysis.txt
echo "$(date '+%F %T') RERUN DONE" >> $P
