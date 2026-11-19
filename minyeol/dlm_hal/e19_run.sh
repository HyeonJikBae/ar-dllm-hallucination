#!/bin/bash
# E12 본 실행: 조건 묶음 하나를 빈 GPU (>=22GB) 에 올리고, 실패하면 재개 (최대 5번).  usage: bash followup_mk/e12_run.sh <이름> "<조건,조건>" <선호 gpu>
cd /NAS/minyeol/ar-dllm-hallucination
P=/home/minyeol/.conda/envs/sae-dream/bin/python; L=followup_mk/logs; n=$1; conds=$2; pref=$3
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L/e19_run.txt; }
freegpu() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' -v p=$pref '$2-$3>=19500{if($1==p){print $1; f=1; exit} else if(o=="") o=$1} END{if(!f && o!="") print o}'; }
for try in 1 2 3 4 5; do
  g=""; until [ -n "$g" ]; do g=$(freegpu); [ -z "$g" ] && sleep 60; done
  log "$n 시작 gpu$g (시도 $try) [$conds]"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g TOKENIZERS_PARALLELISM=false $P followup_mk/e19_lda.py --conds "$conds" --tag $n >> $L/e19_$n.log 2>&1; rc=$?
  [ $rc -eq 0 ] && { log "$n 끝"; exit 0; }
  log "$n 실패 rc$rc, 5분 뒤 재개"; sleep 300
done
log "$n 5번 실패, 멈춤"
