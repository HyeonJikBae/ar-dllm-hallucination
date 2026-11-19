#!/bin/bash
# E52 진행 (로컬 CPU 에서 떼어 실행; GPU 작업은 waitlaunch2 로 빈 GPU 에 띄움)
ROOT=/NAS/minyeol/ar-dllm-hallucination; cd $ROOT
W=${SCRATCH:-/tmp/scratch}/waitlaunch2.sh
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; CPY=/NAS/minyeol/limit/.venv-gfmrag/bin/python; E=followup_mk/e52
cnt() { [ -f $1 ] && wc -l < $1 || echo 0; }
waitlog() { until grep -qE "^DONE|Traceback|Killed" followup_mk/logs/$1.log 2>/dev/null; do sleep 30; done; grep -q "^DONE" followup_mk/logs/$1.log || { echo "FAIL $1"; exit 1; }; }
if [ $(cnt $E/gen_base_s1.jsonl) -lt 2984 ]; then $W 24576 e52_base_s1 "$PY followup_mk/e52_run.py --mode gen --conds base --shard 1 --nshards 2"; fi
until [ $(cnt $E/gen_base_s0.jsonl) -ge 2966 ] && [ $(cnt $E/gen_base_s1.jsonl) -ge 2984 ]; do sleep 30; done
echo "base done $(date)"
$CPY followup_mk/e52_comp.py --dir $E
$W 24576 e52_dla "$PY followup_mk/e52_run.py --mode dla --conds base,brel,rand"; sleep 60; waitlog e52_dla
echo "dla done $(date)"
$CPY followup_mk/e52_subset.py
C=empty_eager,ko_all,rnd_all,ko_20_27,brel,rand,basesub,ko_10_19,ko_0_9
$W 24576 e52_sub_s0 "$PY followup_mk/e52_run.py --mode gen --sub sub --conds $C --shard 0 --nshards 2"; sleep 120
$W 24576 e52_sub_s1 "$PY followup_mk/e52_run.py --mode gen --sub sub --conds $C --shard 1 --nshards 2"
echo "sub launched $(date)"
