#!/bin/bash
# E51 실행기 (daissv3 에서 setsid nohup 으로 띄움).  빈 메모리 >= 24GB 인 GPU 를 기다려 base -> cont tau2, tau4 를 띄운다.
# cont 는 base 와 같은 서버에서만 띄운다 (같은 커널이어야 'own' 이 base 를 그대로 재현).  다른 프로세스는 건드리지 않는다.
cd /NAS/minyeol/ar-dllm-hallucination
P=/home/minyeol/.conda/envs/sae-dream/bin/python; V4=minyeol@163.152.161.200; NEED=24576; LG=followup_mk/logs
ts() { date '+%m-%d %H:%M:%S'; }
free_list() {
  if [ "$1" = v4 ]; then ssh -o BatchMode=yes -o ConnectTimeout=20 $V4 "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
  else nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits; fi | awk -F', ' -v need=$NEED '{ if ($3-$2 >= need) print $1 }'
}
pick() {
  for h in $1; do
    gs=$(free_list $h)
    if [ "$h" = v4 ] && echo "$gs" | grep -qx 2; then echo "v4 2"; return; fi
    g=$(echo "$gs" | head -1); if [ -n "$g" ]; then echo "$h $g"; return; fi
  done
}
launch() {
  if [ "$1" = v4 ]; then ssh -f -n -o BatchMode=yes $V4 "cd /NAS/minyeol/ar-dllm-hallucination && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$2 HF_HUB_OFFLINE=1 setsid nohup $P followup_mk/e51_canvas.py $3 > $LG/$4 2>&1 &"
  else (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$2 HF_HUB_OFFLINE=1 setsid nohup $P followup_mk/e51_canvas.py $3 > $LG/$4 2>&1 &); fi
  echo "$(ts) 띄움 $1 GPU $2: $3 -> $4"
}
done_log() { grep -q '^DONE' $LG/$1 2>/dev/null; }
fail_log() { grep -q 'Traceback' $LG/$1 2>/dev/null; }
# ---- base
if ! done_log e51_base.log; then
  until pk=$(pick "v4 v3"); [ -n "$pk" ]; do sleep 60; done
  set -- $pk; HOST=$1; launch $1 $2 "--phase base" e51_base.log; echo $HOST > $LG/e51_host.txt
  until done_log e51_base.log; do if fail_log e51_base.log; then echo "$(ts) base 실패"; exit 1; fi; sleep 60; done
fi
HOST=$(cat $LG/e51_host.txt); echo "$(ts) base 끝 (서버 $HOST)"
# ---- cont tau 2, 4 (같은 서버)
L2=0; L4=0
done_log e51_t2.log && L2=1; done_log e51_t4.log && L4=1
while [ $L2 = 0 ] || [ $L4 = 0 ]; do
  pk=$(pick "$HOST")
  if [ -n "$pk" ]; then
    set -- $pk
    if [ $L2 = 0 ]; then launch $1 $2 "--phase cont --tau 2" e51_t2.log; L2=1; elif [ $L4 = 0 ]; then launch $1 $2 "--phase cont --tau 4" e51_t4.log; L4=1; fi
    sleep 240; continue
  fi
  sleep 60
done
until done_log e51_t2.log && done_log e51_t4.log; do if fail_log e51_t2.log || fail_log e51_t4.log; then echo "$(ts) cont 실패"; exit 1; fi; sleep 60; done
echo "$(ts) 모두 끝"
