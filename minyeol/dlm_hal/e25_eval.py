"""E25 평가 (PREREG.md, CPU): 변형 Brel c1 (좌표 마스크 + 관계 안 게이트) 전체 규모, 한 번 생성.  첫 생성 = E7, 비교 = E12 의 조밀 Brel c1."""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import grp, mid, NF, OUT, e7_rows, set_S
_, _, F7 = e7_rows(); R = np.array([int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]); n = len(R); ism = mid[R]; isf = ~ism
meta = {}
for l in open(f"{OUT}/e7/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
first = np.array([meta[int(i)]["new_type"] == "correct" for i in R]); typ = np.array([meta[int(i)]["new_type"] for i in R]); g = grp[R]
def load(pat, cond):
    E = {}
    for f in sorted(glob.glob(f"{OUT}/{pat}")):
        for l in open(f, encoding="utf-8"):
            try: d = json.loads(l)
            except Exception: continue
            if d["cond"] == cond: E[d["row"]] = (d["new_type"] == "correct", d.get("n_on"))
    return E
V = load("results_e16_full_G.jsonl", "변형 Brel c1"); D = load("results_e12_[A-Z].jsonl", "Brel c1"); assert len(V) == n and len(D) == n, (len(V), len(D))
var = np.array([V[int(i)][0] for i in R]); dense = np.array([D[int(i)][0] for i in R]); non = np.array([V[int(i)][1] for i in R], float)
SC = np.load(f"{OUT}/e12_sc.npy"); rng = np.random.default_rng(0)
def ci(d, B=4000): bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]) * 100; return [round(float(d.mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
def block(p, ref=first):
    m = np.where(ism)[0]; w = ism & ~ref; c = ism & ref
    return {"1~4/5 이득 [CI]": ci((p.astype(float) - ref.astype(float))[m]), "오답 복구 %": round(100 * float(p[w].mean()), 1), "정답 깨짐 %": round(100 * float((~p)[c].mean()), 1), "5/5 %": round(100 * float(p[isf].mean()), 2),
            "붕괴 복구 %": round(100 * float(p[w & (typ == "broken")].mean()), 1), "entity 복구 %": round(100 * float(p[w & (typ == "entity")].mean()), 1)}
O = {"켜진 차원 평균": round(float(np.nanmean(non)), 1), "한 번 생성 (변형 Brel c1)": block(var), "한 번 생성 (조밀 Brel c1)": block(dense)}
d_ = (var.astype(float) - dense.astype(float))[ism]; O["변형 - 조밀 (1~4/5)"] = ci(d_)
def twopass(redo, use):
    sel = np.zeros(n, bool)
    for gg in range(NF):
        tr, te = (g != gg) & use, (g == gg) & use; bq, bv = 0.0, -1e9
        for q in np.linspace(0, 1, 51):
            th = np.quantile(SC[tr], 1 - q) if q > 0 else np.inf; v = np.where(SC >= th, redo, first)[tr & ism].mean()
            if v > bv + 1e-12: bv, bq = v, q
        th = np.quantile(SC[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = SC[te] >= th
    return {**block(np.where(sel, redo, first)), "다시생성 1~4/5 %": round(100 * float(sel[ism].mean()), 1), "다시생성 5/5 %": round(100 * float(sel[isf].mean()), 1)}
use = np.ones(n, bool); Sset = {i for _, i in set_S()}; notS = np.array([int(i) not in Sset for i in R])
O["두 번 생성 (변형)"] = twopass(var, use); O["두 번 생성 (조밀)"] = twopass(dense, use); O["두 번 생성 (변형, S 제외)"] = twopass(var, notS)
h = O["한 번 생성 (변형 Brel c1)"]; hd = O["한 번 생성 (조밀 Brel c1)"]
O["판정"] = {"H25.1": bool(h["1~4/5 이득 [CI]"][0] >= 3 and h["1~4/5 이득 [CI]"][1] > 0 and h["5/5 %"] >= 97), "H25.2": bool(h["1~4/5 이득 [CI]"][0] - hd["1~4/5 이득 [CI]"][0] >= 1 and h["5/5 %"] > hd["5/5 %"])}
print(json.dumps(O, ensure_ascii=False, indent=1)); json.dump(O, open(f"{OUT}/e25_summary.json", "w"), ensure_ascii=False, indent=1)
