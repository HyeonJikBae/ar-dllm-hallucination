"""서버 간 일치 점검: e12_regen 과 같은 묶음 구성으로 처음 N 묶음을 base 로 돌려 E12 base (daissv3) 와 답을 비교."""
import json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import *
NCH = int(sys.argv[sys.argv.index("--chunks") + 1]) if "--chunks" in sys.argv else 6
_, _, F7 = e7_rows(); R = [int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]
ref = {}
for f in ("results_e12_B.jsonl",):
    for l in open(f"{OUT}/{f}", encoding="utf-8"):
        d = json.loads(l)
        if d["cond"] == "base": ref[d["row"]] = d["answer"]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R}
same = tot = 0; diffs = []
for c, (L, ix) in enumerate(by_length(R, lambda i: ids[i])):
    if c >= NCH: break
    ans = m.generate([ids[i] for i in ix])
    for i, a in zip(ix, ans):
        tot += 1; same += (a == ref[i])
        if a != ref[i]: diffs.append((i, ref[i], a))
print(f"일치 {same}/{tot}", flush=True)
for d in diffs[:5]: print(d, flush=True)
print("DONE", flush=True)
