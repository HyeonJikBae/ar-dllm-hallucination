"""E52 (CPU): base 생성 결과에서 사실별 '최종 경쟁 첫 토큰' 을 정함 (결과 분석 전, DLA 계산용).
규칙: 사실 안 오답 (entity/broken) 문구들의 칸 0 최종 토큰 중 금 첫 토큰이 아닌 것의 최빈값 (동률이면 행 번호가 작은 쪽).  그 사실의 다섯 문구 모두 같은 경쟁 토큰을 씀.
후보가 없는 사실은 -1 (DLA 분석에서 뺌).
usage: python followup_mk/e52_comp.py [--dir followup_mk/e52]"""
import collections, glob, json, sys
sys.path.insert(0, "followup_mk")
import numpy as np
from common import rows, fid, OUT
DIR = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else f"{OUT}/e52"
G = {}
for f in sorted(glob.glob(f"{DIR}/gen_base_s*.jsonl")):
    for l in open(f, encoding="utf-8"): d = json.loads(l); G[d["row"]] = d
byf = collections.defaultdict(list)
for i in sorted(G): byf[fid[i]].append(i)
comp = {}; nf = collections.Counter()
for f_, ix in byf.items():
    c = collections.Counter(G[i]["s0_tok"] for i in ix if G[i]["new_type"] in ("entity", "broken") and G[i]["s0_tok"] != G[i]["gold_id"] and G[i]["s0_tok"] >= 0)
    t = c.most_common(1)[0][0] if c else -1; nf["ok" if t >= 0 else "none"] += 1
    for i in ix: comp[str(i)] = int(t)
json.dump(comp, open(f"{DIR}/comp.json", "w"))
print(len(G), "행", dict(nf))
