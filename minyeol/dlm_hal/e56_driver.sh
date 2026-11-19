#!/bin/bash
# E56/E58a 자동 진행: (1) 남은 base 조각 11/16 을 빈 GPU 에 -> (2) base 전부 끝나면 e56_eval, e58a dirs -> (3) 조향 재생성 2 조각 -> (4) e58a eval.
# GPU: 로컬 daissv3 / daissv4 중 빈 메모리 >= 26000 MiB 인 GPU 만 (server5 는 느려서 쓰지 않음).  다른 사람 작업은 건드리지 않음.
cd /NAS/minyeol/ar-dllm-hallucination
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; CPY=/NAS/minyeol/limit/.venv-gfmrag/bin/python; L=followup_mk/logs
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
freegpu() {  # -> "local N" 또는 "v4 N"
  while true; do
    a=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{f=$3-$2; if (f>=26000) print "local", $1}' | head -1)
    [ -n "$a" ] && { echo "$a"; return; }
    a=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 minyeol@163.152.161.200 "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits" 2>/dev/null | awk -F', ' '{f=$3-$2; if (f>=26000) print "v4", $1}' | head -1)
    [ -n "$a" ] && { echo "$a"; return; }
    sleep 120
  done
}
launch() {  # launch "<args>" <logname>
  g=$(freegpu); h=${g% *}; i=${g#* }
  C="cd /NAS/minyeol/ar-dllm-hallucination && CUDA_VISIBLE_DEVICES=$i HF_HUB_OFFLINE=1 setsid nohup $PY followup_mk/e56_dream.py $1 > $L/$2.log 2>&1 < /dev/null &"
  if [ "$h" = local ]; then bash -c "$C"; else ssh -f -n -o BatchMode=yes minyeol@163.152.161.200 "$C"; fi
  log "launched $1 on $g"; sleep 180   # 메모리가 잡힐 때까지 기다린 뒤 다음 빈 GPU 를 찾음
}
waitdone() { for f in "$@"; do until grep -q "^DONE" $L/$f.log 2>/dev/null; do grep -q Traceback $L/$f.log 2>/dev/null && { log "ERROR in $f"; exit 1; }; sleep 60; done; done; }
until grep -q "^DONE" $L/e56_dream_s7of8.log; do sleep 60; done
launch "--shard 11 --nshard 16" e56_dream_s11of16
waitdone e56_dream_s0 e56_dream_s1of4 e56_dream_s7of8 e56_dream_s3of16 e56_dream_s11of16
log "base 끝"
$CPY followup_mk/e56_eval.py > $L/e56_eval.log 2>&1 || { log "e56_eval 실패"; }
$CPY followup_mk/e58a.py --stage dirs > $L/e58a_dirs.log 2>&1 || { log "dirs 실패"; exit 1; }
log "dirs 끝"; launch "--steer --shard 0 --nshard 2" e56_steer_s0; launch "--steer --shard 1 --nshard 2" e56_steer_s1
waitdone e56_steer_s0 e56_steer_s1
$CPY followup_mk/e58a.py --stage eval > $L/e58a_eval.log 2>&1 || { log "e58a eval 실패"; exit 1; }
log "ALL_DONE"
