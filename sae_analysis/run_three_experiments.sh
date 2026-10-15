#!/usr/bin/env bash
# 실험 1·2·3: Qwen 추출 -> Dream/Qwen 통합 분석 -> Dream 선택 문구 재생성. VS Code를 꺼도 계속 돈다.
#   setsid nohup bash sae_analysis/run_three_experiments.sh > sae_analysis/logs/run_three.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
PYS=/home/minyeol/.conda/envs/sae/bin/python
PYD=/home/minyeol/.conda/envs/sae-dream/bin/python
R=sae_analysis/results_twelve
export PYTHONPATH=sae_analysis:grounded_dream
echo "$(date '+%F %T') qwen extract + dream unified (parallel)"
CUDA_VISIBLE_DEVICES=0 $PYS sae_analysis/qwen_step0_extract.py > sae_analysis/logs/qwen_step0_ext.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES= $PYS sae_analysis/signals_unified.py --model dream --dir sae_analysis/features_dream_step0_ext > $R/report_unified_dream.txt 2> sae_analysis/logs/unified_dream.err
wait $P1
echo "$(date '+%F %T') qwen extract end"
CUDA_VISIBLE_DEVICES= $PYS sae_analysis/signals_unified.py --model qwen --dir sae_analysis/features_qwen_step0_ext > $R/report_unified_qwen.txt 2> sae_analysis/logs/unified_qwen.err
echo "$(date '+%F %T') qwen unified end"
CUDA_VISIBLE_DEVICES=0 $PYD sae_analysis/dream_regen_check.py --signal "확신도 + SAE 20개" --setting all > $R/report_dream_regen.txt 2> sae_analysis/logs/dream_regen.err
echo "$(date '+%F %T') regen end"
echo ALL_DONE
