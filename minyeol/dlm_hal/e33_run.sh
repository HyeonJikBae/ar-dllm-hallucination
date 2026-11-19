#!/bin/bash
# E33/E34 실행: 지정 GPU 의 빈 메모리가 20GB 이상이 될 때까지 기다렸다가 올리고, 실패하면 5 분 뒤 재개 (최대 5 번).  usage: bash followup_mk/e33_run.sh <이름> <gpu> <스크립트와 인자...>
cd /NAS/minyeol/ar-dllm-hallucination
P=/home/minyeol/.conda/envs/sae-dream/bin/python; L=followup_mk/logs; n=$1; g=$2; shift 2
log() { echo "$(date '+%m-%d %H:%M:%S') $(hostname) $*" >> $L/e33_run.txt; }
free() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' -v p=$g '$1==p{print $2-$3}'; }
for try in 1 2 3 4 5; do
  until [ "$(free)" -ge 20000 ]; do sleep 60; done
  log "$n 시작 gpu$g (시도 $try) $*"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false $P "$@" >> $L/$n.log 2>&1; rc=$?
  [ $rc -eq 0 ] && { log "$n 끝"; exit 0; }
  log "$n 실패 rc$rc, 5분 뒤 재개"; sleep 300
done
log "$n 5번 실패, 멈춤"
