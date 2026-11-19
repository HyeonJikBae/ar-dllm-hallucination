"""E32 평가 (PREREG.md, CPU).  usage: python followup_mk/e32_eval.py [--pilot]"""
import collections, json, sys
import numpy as np
OUT = "followup_mk"; PIL = "--pilot" in sys.argv
R = [json.loads(l) for l in open(f"{OUT}/results_e32{'_pilot' if PIL else ''}.jsonl", encoding="utf-8")]
by = collections.defaultdict(dict)
for r in R: by[r["cond"]][r["row"]] = r
e20 = {}
for l in open(f"{OUT}/results_e20.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["cond"] == "정답토큰고정": e20[d["row"]] = d["answer"]
rng = np.random.default_rng(0); NB = 4000
def ci(x):
    x = np.asarray(x, float); bs = x[rng.integers(0, len(x), (NB, len(x)))].mean(1) * 100
    return [round(float(x.mean() * 100), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]
O = {}
t0 = by["정답토큰|tau0"]; O["sanity_tau0_대_E20_답일치"] = f"{sum(t0[i]['answer'] == e20.get(i) for i in t0)}/{len(t0)}"
print("sanity τ=0 대 E20 정답토큰고정 답 일치", O["sanity_tau0_대_E20_답일치"], flush=True)
# ---- 정답 토큰 (오답 행)
TA = (0, 2, 4, 8, 16, 24); base = by["정답토큰|base"]; rows_ = sorted(base)
bw = [i for i in rows_ if base[i]["new_type"] != "correct"]   # 같은 묶음 base 가 오답인 행 (복구 분모)
O["정답토큰_행"] = dict(전체=len(rows_), base오답=len(bw), base정답률=ci([base[i]["new_type"] == "correct" for i in rows_]))
tab = {}
for t in TA:
    c = by[f"정답토큰|tau{t}"]; rec = lambda ix: [c[i]["new_type"] == "correct" for i in ix]
    o = {"복구 (base 오답 행)": ci(rec(bw))}
    for nm in ("broken", "entity"):
        ix = [i for i in bw if c[i]["set"] == nm]; o[f"{nm} 복구"] = ci(rec(ix)) + [len(ix)]
    pc = [i for i in bw if c[i]["pre_committed"]]; npc = [i for i in bw if c[i]["pre_committed"] is False]
    o["칸0 이 τ 전에 이미 확정된 행 복구"] = (ci(rec(pc)) + [len(pc)]) if pc else None; o["아직 마스크였던 행 복구"] = (ci(rec(npc)) + [len(npc)]) if npc else None
    for nm in ("broken", "entity"):
        ix = [i for i in bw if c[i]["set"] == nm]; o[f"{nm} 중 τ 전 확정 비율 %"] = round(100 * float(np.mean([bool(c[i]["pre_committed"]) for i in ix])), 1) if ix else None
    tab[f"τ={t}"] = o; print(f"τ={t}", json.dumps(o, ensure_ascii=False), flush=True)
O["정답토큰_τ별"] = tab
# 비율 rec(16)/rec(0) 부트스트랩
def recv(t, ix): return np.array([by[f"정답토큰|tau{t}"][i]["new_type"] == "correct" for i in ix], float)
def ratio(t, ix):
    a, b = recv(t, ix), recv(0, ix); n = len(ix); idx = rng.integers(0, n, (NB, n)); rr = a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-9)
    return [round(float(a.mean() / max(b.mean(), 1e-9)), 3), round(float(np.percentile(rr, 2.5)), 3), round(float(np.percentile(rr, 97.5)), 3)]
O["비율 rec(16)/rec(0)"] = ratio(16, bw)
half = {}
for nm in ("broken", "entity"):
    ix = [i for i in bw if base[i]["set"] == nm]; r0 = recv(0, ix).mean(); first = None
    for t in TA[1:]:
        if recv(t, ix).mean() <= 0.5 * r0: first = t; break
    half[nm] = dict(rec0=round(100 * float(r0), 1), 절반이하가_되는_첫_τ=first, 곡선=[round(100 * float(recv(t, ix).mean()), 1) for t in TA])
O["유형별 곡선"] = half; print("유형별", json.dumps(half, ensure_ascii=False), flush=True)
r16, r0 = recv(16, bw).mean(), recv(0, bw).mean()
mono = all(recv(TA[j + 1], bw).mean() <= recv(TA[j], bw).mean() + 0.02 for j in range(len(TA) - 1))
hb, he = half["broken"]["절반이하가_되는_첫_τ"], half["entity"]["절반이하가_되는_첫_τ"]
O["판정"] = {"H32.1": bool(r16 <= 0.5 * r0 and mono), "H32.1_세부": dict(rec0=round(100 * float(r0), 1), rec16=round(100 * float(r16), 1), 대체로_감소=mono),
            "H32.2": bool(hb is not None and (he is None or hb < he)), "H32.2_세부": dict(붕괴=hb, entity=he)}
# ---- 오답 토큰 (정답 행)
bc = by["오답토큰|base"]; cr = sorted(bc); keep = [i for i in cr if bc[i]["new_type"] == "correct"]; wt = {}
for t in (0, 4, 8, 16):
    c = by[f"오답토큰|tau{t}"]; pc = [i for i in keep if c[i]["pre_committed"]]; npc = [i for i in keep if c[i]["pre_committed"] is False]
    wt[f"τ={t}"] = {"정답 유지 (base 정답 행)": ci([c[i]["new_type"] == "correct" for i in keep]), "칸0 τ 전 확정 비율 %": round(100 * len(pc) / max(len(keep), 1), 1),
                    "τ 전 확정 행 유지": (ci([c[i]["new_type"] == "correct" for i in pc]) + [len(pc)]) if pc else None, "마스크였던 행 유지": (ci([c[i]["new_type"] == "correct" for i in npc]) + [len(npc)]) if npc else None}
    print(f"오답토큰 τ={t}", json.dumps(wt[f"τ={t}"], ensure_ascii=False), flush=True)
O["오답토큰_τ별"] = wt; O["오답토큰_행"] = dict(전체=len(cr), base정답=len(keep))
print("판정", json.dumps(O["판정"], ensure_ascii=False), flush=True)
json.dump(O, open(f"{OUT}/e32_summary{'_pilot' if PIL else ''}.json", "w"), ensure_ascii=False, indent=1); print("DONE")
