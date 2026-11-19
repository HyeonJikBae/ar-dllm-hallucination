#!/bin/bash
cd /NAS/minyeol/ar-dllm-hallucination
bash followup_mk/run_e14.sh followup_mk/logs/e14_pilot.log followup_mk/e14_remask.py --limit 20 || { echo "파일럿 실패" >> followup_mk/logs/e14_runs.txt; exit 1; }
/NAS/minyeol/limit/.venv-gfmrag/bin/python followup_mk/report_e14.py --pilot > followup_mk/logs/report_e14_pilot.txt 2>&1
bash followup_mk/run_e14.sh followup_mk/logs/e14_main.log followup_mk/e14_remask.py || { echo "본 실행 실패" >> followup_mk/logs/e14_runs.txt; exit 1; }
/NAS/minyeol/limit/.venv-gfmrag/bin/python followup_mk/report_e14.py > followup_mk/logs/report_e14.txt 2>&1
echo "$(date '+%m-%d %H:%M:%S') 체인 끝" >> followup_mk/logs/e14_runs.txt
