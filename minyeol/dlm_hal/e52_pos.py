"""E52 준비 (CPU): 1~4/5 행의 프롬프트 토큰, 질문 끝, 주어 끝 위치 (common.positions) 와 경로 라벨 (Qwen 같은 문구 정오) 을 저장.
usage: python followup_mk/e52_pos.py   -> followup_mk/e52/pos.json"""
import json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import *
Q = {}
for l in open(f"{D_}/3_classification/full/qwen_final_full.jsonl", encoding="utf-8"):
    d = json.loads(l); Q[(d["sample_id"], d["version"])] = d
out = {}
for i in np.where(mid)[0]:
    i = int(i); toks, ql, se = positions(i); q = Q[(rows[i]["sample_id"], rows[i]["version"])]
    out[i] = {"ids": toks, "ql": ql, "se": se, "qwen_correct": bool(q["correct"]), "qwen_type": q["type3"] if not q["correct"] else "correct"}
json.dump(out, open(f"{OUT}/e52/pos.json", "w"))
print(len(out), "subj None", sum(v["se"] is None for v in out.values()))
