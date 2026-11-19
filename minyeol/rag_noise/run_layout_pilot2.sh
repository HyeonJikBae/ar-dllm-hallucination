#!/bin/bash
# v10.1 새 틀 파일럿 + v13 코드 확인 (GPU 1).
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_layout_pilot2.txt; : > $P
export CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True
( $PY -u gen_fmt.py --model qwen --data layout_pilot2 --fmt base --bs 16 > logs/fmt_qwen_pilot2.log 2>&1; echo "$(date '+%F %T') end qwen (exit $?)" >> $P ) &
( $PY -u gen_fmt.py --model dream --data layout_pilot2 --fmt base --bs 10 --conds S_e,S_l,L1,L1_sw,L4,L4_sw,L7,L7_sw > logs/fmt_dream_pilot2.log 2>&1; echo "$(date '+%F %T') end dream (exit $?)" >> $P ) &
( $PY -u check_var.py > logs/check_var.log 2>&1; echo "$(date '+%F %T') end check_var (exit $?)" >> $P ) &
wait; echo "$(date '+%F %T') ALL DONE" >> $P
