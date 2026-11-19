"""E8 (PREREG.md): 디코딩 방식·예산.  행 S (1,510).  usage: python followup_mk/e8_decode.py [--limit N]"""
import json, sys, time
import numpy as np
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}; m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}
CONDS = [("base", dict()), ("entropy", dict(alg="entropy")), ("maskgit_plus", dict(alg="maskgit_plus")), ("origin", dict(alg="origin")),
         ("canvas16", dict(canvas=16, steps=16)), ("canvas64", dict(canvas=64, steps=64))]
fo = open(f"{OUT}/results_e8{'_pilot' if LIM else ''}.jsonl", "w", encoding="utf-8")
for cname, kw in CONDS:
    t0 = time.time()
    for bi, (L, ix) in enumerate(by_length([i for _, i in S], lambda i: ids[i])):
        ans = m.generate([ids[i] for i in ix], seed=1234 + bi if cname == "origin" else None, **kw)
        for i, a in zip(ix, ans): fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "k": int(k[i]), "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n")
    fo.flush(); print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
