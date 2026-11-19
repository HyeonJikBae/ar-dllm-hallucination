#!/bin/bash
# usage: bash followup_mk/run_e14.sh <log> <python args...>   빈 메모리 >= 22GB GPU 기다렸다 실행, OOM/실패 시 5번까지 재시도
LOG=$1; shift
PY=/home/minyeol/.conda/envs/sae-dream/bin/python
for a in 1 2 3 4 5; do
  while true; do
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=22000{print $1; exit}')
    [ -n "$G" ] && break; sleep 60
  done
  echo "$(date '+%m-%d %H:%M:%S') 시작 gpu$G (시도 $a): $*" >> followup_mk/logs/e14_runs.txt
  CUDA_VISIBLE_DEVICES=$G $PY "$@" > $LOG 2>&1; rc=$?
  echo "$(date '+%m-%d %H:%M:%S') 끝 rc$rc: $*" >> followup_mk/logs/e14_runs.txt
  [ $rc -eq 0 ] && exit 0
  sleep 90
done
exit 1
