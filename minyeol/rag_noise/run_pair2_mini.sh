#!/bin/bash
cd /NAS/minyeol/ar-dllm-hallucination/rag_noise; PY=/home/minyeol/.conda/envs/sae-dream/bin/python; export CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True
C=S_e,S_l,L1,L1_sw,L7,L7_sw

$PY -u gen_pair2.py --model llama2 --data layout_pilot --conds $C --bs 4 > logs/mini_llama2.log 2>&1

echo DONE > logs/mini_pair2.done
