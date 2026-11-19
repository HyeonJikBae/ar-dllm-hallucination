#!/bin/bash
# run_sweep.sh 가 끝나면 채점 (Qwen vs Dream s16 + 병렬도 sweep + 전제 감사) 과 궤적 분석 -> logs/analysis.txt
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise
PY=/home/minyeol/.conda/envs/sae-dream/bin/python
until grep -q "ALL DONE" logs/PROGRESS_sweep.txt; do sleep 60; done
{ $PY grade_anchor.py --sweep; echo; echo "===== 궤적 (H3) ====="; $PY trace_anchor.py; } > logs/analysis.txt 2>&1
echo "ANALYSIS DONE $(date '+%F %T')" >> logs/analysis.txt
