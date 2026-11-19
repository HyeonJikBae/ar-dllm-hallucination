#!/bin/bash
# E52 pilot: 8 묶음, gen -> comp -> dla -> analyze (pilot 디렉터리)
set -e
cd /NAS/minyeol/ar-dllm-hallucination
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; CPY=/NAS/minyeol/limit/.venv-gfmrag/bin/python
$PY followup_mk/e52_run.py --mode gen --conds base,empty_eager,ko_all,rnd_all,ko_20_27,brel,rand --limit 8
$CPY followup_mk/e52_comp.py --dir followup_mk/e52_pilot
$PY followup_mk/e52_run.py --mode dla --conds base,brel,rand --limit 8
$CPY followup_mk/e52_analyze.py --dir followup_mk/e52_pilot --out followup_mk/e52_summary_pilot.json > followup_mk/logs/e52_pilot_analyze.log 2>&1
echo PILOT_DONE
