#!/bin/bash
# GPU 빈 메모리가 MINF MiB 이상인 GPU (daissv3 0,1 / daissv4 0,1,2) 를 찾을 때까지 60 초마다 확인한 뒤 CMD 를 그 서버에서 떼어 실행.
# usage: bash followup_mk/e5354_launch.sh NAME MINF_MiB "CMD"   (CMD 는 repo 루트 기준, CUDA_VISIBLE_DEVICES 는 여기서 붙임)
NAME=$1; MINF=$2; CMD=$3; V4=minyeol@163.152.161.200
cd /NAS/minyeol/ar-dllm-hallucination
while true; do
  for g in 1 0; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g); if [ "$f" -ge "$MINF" ]; then H=local; G=$g; break 2; fi; done
  for g in 2 0 1; do f=$(ssh -o BatchMode=yes -o ConnectTimeout=10 $V4 "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g" 2>/dev/null); if [ -n "$f" ] && [ "$f" -ge "$MINF" ]; then H=v4; G=$g; break 2; fi; done
  sleep 60
done
echo "$(date '+%F %T') $NAME host=$H gpu=$G" >> followup_mk/logs/e5354_launch.txt
SC="mkdir -p ${SCRATCH:-/tmp/scratch}/e53; cd /NAS/minyeol/ar-dllm-hallucination && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup bash -c '$CMD' > followup_mk/logs/$NAME.log 2>&1 &"
if [ $H = local ]; then bash -c "$SC"; else ssh -f -n -o BatchMode=yes $V4 "$SC"; fi
