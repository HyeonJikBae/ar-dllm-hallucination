"""E52 (CPU): 주의 막기·조향 생성용 부분 집합 (결과 보기 전, 계산 시간 때문에).  전체 base 에서 오답 (entity/broken) 이면서 step 2 에 칸 0 이 아직 마스크인 행 전부
+ 같은 조건의 정답 행 무작위 800 (seed 52).  -> followup_mk/e52/sub/rows.json"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import OUT
G = {}
for f in sorted(glob.glob(f"{OUT}/e52/gen_base_s*.jsonl")):
    for l in open(f, encoding="utf-8"): d = json.loads(l); G[d["row"]] = d
w = sorted(i for i, d in G.items() if d["new_type"] in ("entity", "broken") and d["rank"][2] >= 0)
c = sorted(i for i, d in G.items() if d["new_type"] == "correct" and d["rank"][2] >= 0)
cs = sorted(int(x) for x in np.random.default_rng(52).choice(c, min(800, len(c)), replace=False))
os.makedirs(f"{OUT}/e52/sub", exist_ok=True); json.dump(sorted(w + cs), open(f"{OUT}/e52/sub/rows.json", "w"))
print(f"base 행 {len(G)}, 오답 step2 마스크 {len(w)}, 정답 step2 마스크 {len(c)} 중 {len(cs)}")
