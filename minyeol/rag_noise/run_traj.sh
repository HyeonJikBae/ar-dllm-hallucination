#!/bin/bash
# Dream 생성 궤적 (PREREG_poison.md v5), GPU 0.  사용: setsid nohup bash run_traj.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs; P=logs/PROGRESS_traj.txt
echo "$(date '+%F %T') start traj dream bindpatch" >> $P
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/minyeol/.conda/envs/sae-dream/bin/python -u traj_bind.py --data bindpatch > logs/traj_dream_bindpatch.log 2>&1
echo "$(date '+%F %T') end (exit $?)" >> $P
