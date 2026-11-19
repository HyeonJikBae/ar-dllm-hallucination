#!/bin/bash
# 연결 정보 / 최종 답 분리 patching (PREREG_poison.md v4): 파일럿 -> 본 실행, 각각 Qwen -> Dream, GPU 0.
# 사용: setsid nohup bash run_bindpatch.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_bindpatch.txt
for D in bindpatch_pilot bindpatch; do  # v4.1
  for m in qwen dream; do
    echo "$(date '+%F %T') start $m $D" >> $P
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u patch_bind.py --model $m --data $D > logs/patch_${m}_$D.log 2>&1
    rc=$?; echo "$(date "+%F %T") end $m $D (exit $rc)" >> $P
  done
done
echo "$(date '+%F %T') ALL DONE" >> $P
