#!/bin/bash
# E13 대기열: 빈 메모리 >= 22GB GPU 를 기다려 파일럿 -> 본 실행.  실패 시 5 번까지 재시도 (재개 가능).
cd /NAS/minyeol/ar-dllm-hallucination
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; LOG=followup_mk/logs/e13_queue.txt
pick() { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=22528{print $1; exit}'); [ -n "$g" ] && { echo $g; return; }; sleep 60; done; }
run() { local tag=$1; shift; for t in 1 2 3 4 5; do g=$(pick); echo "$(date '+%m-%d %H:%M:%S') $tag 시작 gpu$g (시도 $t)" >> $LOG
  CUDA_VISIBLE_DEVICES=$g $PY followup_mk/e13_capture.py "$@" > followup_mk/logs/e13_$tag.log 2>&1; rc=$?
  if [ $rc -eq 0 ] && grep -q '^DONE' followup_mk/logs/e13_$tag.log; then echo "$(date '+%m-%d %H:%M:%S') $tag 끝" >> $LOG; return 0; fi
  cp followup_mk/logs/e13_$tag.log followup_mk/logs/e13_$tag.fail$t.log; echo "$(date '+%m-%d %H:%M:%S') $tag 실패 rc$rc" >> $LOG; sleep 30; done; return 1; }
if [ "$1" == "pilot" ]; then run pilot --pilot; else run main; fi
