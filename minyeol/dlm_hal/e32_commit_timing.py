"""E32 (PREREG.md): 결정이 굳는 시점.  step τ 에 답 칸 0 을 공여 토큰으로 바꾸고 고정.  공여·행은 E20 (results_e20.jsonl) 그대로, 묶음 구성도 E20 과 같게.
usage: python followup_mk/e32_commit_timing.py [--limit N]"""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import dloop_p6 as dl
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy")
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r"))
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
tok0 = {int(r): int(committed[j, 0]) for j, r in enumerate(R7)}; t0_e7 = {int(r): int(lastm[j, 0]) for j, r in enumerate(R7)}
dc_, dw_ = {}, {}
for l in open(f"{OUT}/results_e20.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["cond"] == "정답토큰고정": dc_[d["row"]] = d["donor"]
    if d["cond"] == "오답토큰고정" and d["set"] == "correct": dw_[d["row"]] = d["donor"]
S = set_S(); st = {i: s for s, i in S}
GR = {"정답토큰": ([i for s, i in S if s in ("broken", "entity") and i in dc_], dc_, (0, 2, 4, 8, 16, 24)),
      "오답토큰": ([i for s, i in S if s == "correct" and i in dw_], dw_, (0, 4, 8, 16))}
if LIM: GR = {g: (rs[:LIM], dn, ts) for g, (rs, dn, ts) in GR.items()}
print({g: len(rs) for g, (rs, _, _) in GR.items()}, flush=True)
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}
fn = f"{OUT}/results_e32{'_pilot' if LIM else ''}.jsonl"; have = set()
if os.path.exists(fn):
    for l in open(fn, encoding="utf-8"):
        try: d = json.loads(l); have.add((d["cond"], d["row"]))
        except Exception: pass
fo = open(fn, "a", encoding="utf-8")
for g, (rs, dn, taus) in GR.items():
    conds = [f"{g}|base"] + [f"{g}|tau{t}" for t in taus]; t0 = time.time()
    for L, ix in by_length(rs, lambda i: ids[i]):
        for c in conds:
            if all((c, i) in have for i in ix): continue
            if c.endswith("|base"): ans, stt = dl.generate(m, [ids[i] for i in ix]); tau = None
            else:
                tau = int(c.split("tau")[1]); ans, stt = dl.generate(m, [ids[i] for i in ix], inject=[tok0[dn[i]] for i in ix], inject_step=tau)
            for j, (i, a) in enumerate(zip(ix, ans)):
                fo.write(json.dumps(dict(cond=c, group=g, set=st[i], row=i, k=int(k[i]), tau=tau, answer=a, new_type=m.grade(i, a), donor=dn[i], donor_tok=m.tok.decode([tok0[dn[i]]]),
                                         pre_committed=(stt["pre_committed"][j] if tau is not None else None), e7_t0=t0_e7.get(i)), ensure_ascii=False) + "\n")
        fo.flush()
    print(f"  {g}: {len(rs)} 행 x {len(conds)} 조건 {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
