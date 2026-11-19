#!/bin/bash
# A, B 가 실제로 생성 중 (결과 파일에 줄이 생김) 일 때 C 를, C 가 생성 중일 때 D 를 올린다 (동시 적재 OOM 피하기).
cd /NAS/minyeol/ar-dllm-hallucination/followup_mk
nl() { [ -f results_e16_$1.jsonl ] && wc -l < results_e16_$1.jsonl || echo 0; }
until [ "$(nl A)" -gt 0 ] && [ "$(nl B)" -gt 0 ]; do sleep 30; done
nohup bash e16_run.sh C "정적+동적 B c4,정적+동적 Brel c1,정적+동적 Brel c2" 1 > /dev/null 2>&1 &
until [ "$(nl C)" -gt 0 ]; do sleep 30; done
nohup bash e16_run.sh D "변형 Brel c1,변형 Brel c2,무작위 B c2" 1 > /dev/null 2>&1 &
