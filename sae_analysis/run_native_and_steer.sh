#!/usr/bin/env bash
# (A) Dream 고유 확신도 재측정 -> 통합 분석, (B) 조향(steering). VS Code를 꺼도 계속 돈다.
#   setsid nohup bash sae_analysis/run_native_and_steer.sh > sae_analysis/logs/run_native_steer.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PYS=/home/minyeol/.conda/envs/sae/bin/python; PYD=/home/minyeol/.conda/envs/sae-dream/bin/python
R=sae_analysis/results_twelve; export PYTHONPATH=sae_analysis:grounded_dream
echo "$(date '+%F %T') steer smoke"
CUDA_VISIBLE_DEVICES=0 $PYD sae_analysis/dream_steer.py --smoke > sae_analysis/logs/steer_smoke.log 2>&1
echo "$(date '+%F %T') native trace"
CUDA_VISIBLE_DEVICES=0 $PYD sae_analysis/dream_native_trace.py > sae_analysis/logs/native_trace.log 2>&1
$PYS sae_analysis/dream_native_features.py > sae_analysis/logs/native_features.log 2>&1
echo "$(date '+%F %T') native unified"
CUDA_VISIBLE_DEVICES= $PYS sae_analysis/signals_unified.py --model dream --dir sae_analysis/features_dream_step0_ext --native $R/dream_native/features.npz --tag _native > $R/report_unified_dream_native.txt 2> sae_analysis/logs/unified_dream_native.err
echo "$(date '+%F %T') steer full"
CUDA_VISIBLE_DEVICES=0 $PYD sae_analysis/dream_steer.py > $R/report_steer.txt 2> sae_analysis/logs/steer_full.err
echo "$(date '+%F %T') steer end"; echo ALL_DONE
