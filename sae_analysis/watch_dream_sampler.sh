#!/usr/bin/env bash
# dream_sampler_compare.py 생성 작업을 지켜보다가, 설정 하나가 끝날 때마다 평가 보고서를 파일로 남긴다 (CPU만 사용).
#   setsid nohup bash sae_analysis/watch_dream_sampler.sh > sae_analysis/logs/watch_dream_sampler.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
D=sae_analysis/results_twelve/dream_sampler
PY=/home/minyeol/.conda/envs/sae-dream/bin/python
rep() { PYTHONPATH=sae_analysis:grounded_dream:sae $PY sae_analysis/dream_sampler_compare.py --eval-only > "$D/report_$1.txt" 2>&1; echo "$(date '+%F %T') report_$1.txt"; }
for cfg in entropy_s64 origin_s64 entropy_s32 origin_s64_seed1 topk_margin_s32 maskgit_plus_s32; do
  until [ "$(wc -l < $D/$cfg.jsonl 2>/dev/null || echo 0)" -ge 5000 ]; do
    pgrep -f "python sae_analysis/dream_sampler_compare" >/dev/null || { echo "$(date '+%F %T') 생성 작업이 종료됨 (현재 $cfg: $(wc -l < $D/$cfg.jsonl 2>/dev/null || echo 0)/5000)"; rep final; exit 0; }
    sleep 60
  done
  rep "$cfg"
done
rep final
echo "$(date '+%F %T') ALL_DONE"
