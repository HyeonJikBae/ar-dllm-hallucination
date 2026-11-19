#!/bin/bash
# 야간 묶음: GPU (20GB 이상 빈 곳) 에 E7, E8, E9 를 파일럿 -> 본 실행 순으로 올리고, 끝나면 CPU 분석과 MORNING.md 작성.
cd /NAS/minyeol/ar-dllm-hallucination
P=/home/minyeol/.conda/envs/sae-dream/bin/python; PC=/NAS/minyeol/limit/.venv-gfmrag/bin/python; L=followup_mk/logs; mkdir -p $L
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L/run_all.txt; }
freegpu() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '$2-$3>=22000{print $1; exit}'; }
job() {  # job <이름> <스크립트> <파일럿 limit>
  n=$1; s=$2; lim=$3
  for try in 1 2 3 4 5; do
    g=""; until [ -n "$g" ]; do g=$(freegpu); [ -z "$g" ] && sleep 60; done
    log "$n 파일럿 시작 gpu$g (시도 $try)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g TOKENIZERS_PARALLELISM=false $P followup_mk/$s --limit $lim > $L/${n}_pilot.log 2>&1; rc=$?
    if [ $rc -ne 0 ]; then grep -q "OutOfMemory\|CUDA out of memory" $L/${n}_pilot.log && { log "$n 파일럿 메모리 부족, 재시도"; sleep 300; continue; }; log "$n 파일럿 실패 rc$rc (본 실행 안 함)"; return; fi
    log "$n 파일럿 통과"
    if [ "$n" = "e7" ]; then $PC followup_mk/e10_detect.py --pilot > $L/e10_pilot.log 2>&1 && log "e10 파일럿 통과" || log "e10 파일럿 실패 (e10_pilot.log 확인)"; fi
    g=""; until [ -n "$g" ]; do g=$(freegpu); [ -z "$g" ] && sleep 60; done
    log "$n 본 실행 시작 gpu$g"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g TOKENIZERS_PARALLELISM=false $P followup_mk/$s > $L/${n}.log 2>&1; rc=$?
    [ $rc -eq 0 ] && { log "$n 본 실행 끝"; return; }
    cp $L/${n}.log $L/${n}.fail$try.log; log "$n 본 실행 실패 rc$rc, 재시도 (e7 은 이어서 함)"; sleep 300
  done
}
log "시작"
job e7 e7_capture.py 20 & p7=$!; sleep 420
job e9 e9_steer.py 12 & p9=$!; sleep 420
job e8 e8_decode.py 12 & p8=$!
wait $p7 $p9 $p8
until grep -q "^e6" $L/rc.txt 2>/dev/null; do sleep 60; done
$PC followup_mk/report_e6_e8_e9.py > $L/report_e6_e8_e9.txt 2>&1; log "보고 E6 E8 E9 rc$?"
$PC followup_mk/e10_detect.py > $L/e10.log 2>&1; log "E10 rc$?"
{ echo "# 야간 결과 ($(date '+%F %H:%M'))"; echo; echo '## 실행 기록'; echo '```'; cat $L/run_all.txt; echo '```'; echo; echo '## E6, E8, E9'; echo '```'; cat $L/report_e6_e8_e9.txt; echo '```'; echo; echo '## E10 감지'; echo '```'; grep -v -i warn $L/e10.log | tail -60; echo '```'; } > followup_mk/MORNING.md
log "끝 (MORNING.md)"
