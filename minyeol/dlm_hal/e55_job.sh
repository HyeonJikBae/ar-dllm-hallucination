#!/bin/bash
# usage: e55_job.sh LOG python-args...   (repo 루트에서; 이 서버에서 빈 메모리 >= NEED MiB 인 GPU 를 골라 실행, 메모리 부족으로 죽으면 기다렸다 이어 돌림)
NEED=${NEED:-24576}; LOG=$1; shift
cd /NAS/minyeol/ar-dllm-hallucination
for attempt in $(seq 1 200); do
  while true; do
    G=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ' | awk -F, -v n=$NEED -v ex="$EXCL" '{f=$3-$2; if (f>=n && index(" "ex" ", " "$1" ")==0) {print $1" "f}}' | sort -k2 -nr | head -1 | cut -d' ' -f1)
    [ -n "$G" ] && break; sleep 30
  done
  echo "=== attempt $attempt GPU $G $(date +%T)" >> $LOG
  CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/minyeol/.conda/envs/sae-dream/bin/python "$@" >> $LOG 2>&1
  rc=$?
  tail -3 $LOG | grep -q "^DONE" && { echo "=== finished $(date +%T)" >> $LOG; exit 0; }
  if tail -5 $LOG | grep -q "OutOfMemoryError\|OOM 반복"; then sleep 90; continue; fi
  echo "=== failed rc=$rc" >> $LOG; exit $rc
done
