#!/bin/bash
# E53: ana 수집이 끝나면 렌즈 적용 (GPU 1, 작은 메모리) -> 분석 (CPU)
cd /NAS/minyeol/ar-dllm-hallucination
until grep -q "^DONE" followup_mk/logs/e53_capture_ana.log || grep -q Traceback followup_mk/logs/e53_capture_ana.log; do sleep 30; done
grep -q Traceback followup_mk/logs/e53_capture_ana.log && exit 1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /home/minyeol/.conda/envs/sae-dream/bin/python followup_mk/e53_apply.py > followup_mk/logs/e53_apply.log 2>&1 && \
/NAS/minyeol/limit/.venv-gfmrag/bin/python followup_mk/e53_analyze.py > followup_mk/logs/e53_analyze.log 2>&1
echo "AFTER_DONE $?" >> followup_mk/logs/e53_analyze.log
