#!/bin/bash
# E52 전체: bash followup_mk/e52_full.sh SHARD NSHARDS
# shard 0 은 base 뒤에 다른 shard 들의 base 가 끝나길 기다려 comp + dla (전체 행, 한 번 forward) 를 돌린 뒤 나머지 생성 조건으로.
set -e
cd /NAS/minyeol/ar-dllm-hallucination
S=$1; N=$2; PY=/home/minyeol/.conda/envs/sae-dream/bin/python; CPY=/NAS/minyeol/limit/.venv-gfmrag/bin/python
$PY followup_mk/e52_run.py --mode gen --conds base --shard $S --nshards $N
touch followup_mk/e52/base_done_s$S
if [ "$S" = "0" ]; then
  for q in $(seq 0 $((N-1))); do until [ -f followup_mk/e52/base_done_s$q ]; do sleep 30; done; done
  $CPY followup_mk/e52_comp.py --dir followup_mk/e52
  $PY followup_mk/e52_run.py --mode dla --conds base,brel,rand --shard 0 --nshards 1
  touch followup_mk/e52/dla_done
fi
$PY followup_mk/e52_run.py --mode gen --conds empty_eager,ko_all,rnd_all,brel,rand,ko_20_27,ko_10_19,ko_0_9,rnd_20_27,rnd_10_19,rnd_0_9 --shard $S --nshards $N
echo FULL_DONE_$S
