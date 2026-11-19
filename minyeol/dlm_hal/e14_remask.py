"""E14 (PREREG.md, H14.2): 답 칸 다시 가리기 (T2M LowProb) 와 흔들리는 답 칸 확정 미루기.  행 S (1,510), E9 와 같은 배치 구성, base 도 같은 고리로.
usage: python followup_mk/e14_remask.py [--limit N]"""
import json, sys, time
import numpy as np
sys.path.insert(0, "followup_mk")
from common import *
import dloop
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}
CONDS = [("base", dict(mode="orig")), ("T2M tau0.3", dict(mode="t2m", tau=0.3, cmax=2)), ("T2M tau0.5", dict(mode="t2m", tau=0.5, cmax=2)), ("흔들림 미룸", dict(mode="defer", defer_k=2))]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}
fo = open(f"{OUT}/results_e14{'_pilot' if LIM else ''}.jsonl", "w", encoding="utf-8")
for cname, kw in CONDS:
    t0 = time.time()
    for L, ix in by_length([i for _, i in S], lambda i: ids[i]):
        ans, stt = dloop.generate(m, [ids[i] for i in ix], **kw)
        for j, (i, a) in enumerate(zip(ix, ans)):
            fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "k": int(k[i]), "answer": a, "new_type": m.grade(i, a), "n_remask": stt["n_remask"][j], "n_defer_steps": stt["n_defer_steps"][j]}, ensure_ascii=False) + "\n")
    fo.flush(); print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
