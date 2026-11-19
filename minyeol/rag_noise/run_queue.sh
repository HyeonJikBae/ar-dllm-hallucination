#!/bin/bash
# 작업 큐.  usage: setsid nohup bash run_queue.sh <큐 파일> "<슬롯별 GPU, 예: 0 0 1 1 1>" > /dev/null 2>&1 < /dev/null &
# 큐 파일 한 줄 = 한 작업 (# 로 시작하면 건너뜀).  슬롯마다 다음 줄을 가져가 실행.  실패하면 300 초 뒤 최대 3 번 재시도 (gen_* 는 이어 쓰기).
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; Q=$1; SLOTS=$2; L=$Q.lock; D=$Q.idx; P=logs/PROGRESS_$(basename $Q).txt; [ -f $D ] || echo 0 > $D; mkdir -p logs/q
PY=/home/minyeol/.conda/envs/sae-dream/bin/python
worker() { local gpu=$1 name=$2
  while :; do
    n=$(flock $L bash -c "i=\$(cat $D); echo \$((i+1)) > $D; echo \$i")
    total=$(wc -l < $Q); [ "$n" -ge "$total" ] && break
    line=$(sed -n "$((n+1))p" $Q); case "$line" in ""|\#*) continue;; esac
    for try in 1 2 3; do
      echo "$(date '+%F %T') [$name gpu$gpu] start #$n try$try: $line" >> $P
      CUDA_VISIBLE_DEVICES=$gpu PYTORCH_ALLOC_CONF=expandable_segments:True bash -c "${line//python/$PY -u}" > logs/q/$(basename $Q)_$n.log 2>&1; rc=$?
      echo "$(date '+%F %T') [$name gpu$gpu] end #$n (exit $rc)" >> $P
      [ $rc -eq 0 ] && break; sleep 300
    done
  done
  echo "$(date '+%F %T') [$name] 슬롯 끝" >> $P
}
i=0; for g in $SLOTS; do worker $g s$i & i=$((i+1)); done; wait; echo "$(date '+%F %T') ALL DONE" >> $P
