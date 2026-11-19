#!/bin/bash
# v10 파일럿.  시작: setsid nohup bash run_layout_pilot.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_layout_pilot.txt; : > $P
run() { echo "$(date '+%F %T') start $1" >> $P; CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u gen_fmt.py --model $1 --data layout_pilot --fmt base --bs 16 > logs/fmt_$1_pilot.log 2>&1; echo "$(date '+%F %T') end $1 (exit $?)" >> $P
        CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u gen_fmt.py --model $1 --data layout_pilot --fmt base --bs 1 --limit 3 --out results/$1_layout_pilot_bs1check.jsonl > logs/fmt_$1_pilot_bs1.log 2>&1; echo "$(date '+%F %T') end $1 bs1 (exit $?)" >> $P; }
run qwen & run dream & wait
echo "$(date '+%F %T') ALL DONE" >> $P
