#!/bin/bash
# A*/D* 본 실행 (data/musique_poison_star.jsonl x As / As_sw / Ds / Ds_sw): D 부분 실행이 끝나면 Qwen -> Dream (steps 16), 배치 1, GPU 0.
# 셋째 Dream 프로세스를 같이 올리면 GPU 0 메모리가 모자라서 D 부분이 끝날 때까지 기다린다.  시작: setsid nohup bash run_poison_star.sh > /dev/null 2>&1 < /dev/null &
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; mkdir -p logs
PY=/home/minyeol/.conda/envs/sae-dream/bin/python; P=logs/PROGRESS_poison_star.txt
until grep -q "ALL DONE" logs/PROGRESS_poison_mainD.txt 2>/dev/null; do sleep 60; done
for m in qwen dream; do
  echo "$(date '+%F %T') start $m" >> $P
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u gen.py --model $m --data poison_star --tag poisonstar --bs 1 > logs/gen_${m}_poisonstar.log 2>&1
  rc=$?; echo "$(date "+%F %T") end $m (exit $rc)" >> $P
done
echo "$(date '+%F %T') ALL DONE" >> $P
