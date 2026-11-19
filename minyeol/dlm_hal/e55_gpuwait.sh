#!/bin/bash
# 빈 메모리가 NEED MiB 이상인 GPU 를 찾을 때까지 기다린다.  출력: "local N" 또는 "v4 N".  EXCL 에 "local:1 v4:2" 처럼 뺄 GPU.
NEED=${NEED:-24576}
while true; do
  for line in $(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' '); do
    IFS=, read i u t <<< "$line"; f=$((t-u)); [[ " $EXCL " == *" local:$i "* ]] && continue
    if [ $f -ge $NEED ]; then echo "local $i $f"; exit 0; fi
  done
  for line in $(ssh -o BatchMode=yes -o ConnectTimeout=10 minyeol@163.152.161.200 "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits" 2>/dev/null | tr -d ' '); do
    IFS=, read i u t <<< "$line"; f=$((t-u)); [[ " $EXCL " == *" v4:$i "* ]] && continue
    if [ $f -ge $NEED ]; then echo "v4 $i $f"; exit 0; fi
  done
  sleep 30
done
