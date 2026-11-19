#!/bin/bash
# 빈 메모리 24GB 이상인 로컬 GPU 가 생기면 샘플러 하나씩 띄움 (먼저 파일럿 1 회로 base 재현 확인은 생략: 기존 bs1 과 같은 gen.prompt 사용)
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; PY=/home/minyeol/.conda/envs/sae-dream/bin/python
for S in entropy:16 topk_margin:16 entropy:8; do
  while true; do
    G=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '$3-$2>24000{print $1; exit}')
    [ -n "$G" ] && break; sleep 120
  done
  echo "$(date) launch $S on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY traj_t2.py --sampler $S > logs/traj_t2_${S/:/_s}.log 2>&1
done
echo "$(date) ALL DONE"
