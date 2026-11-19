"""E33 평가 (PREREG.md, CPU): 전체 규모 3-shot, 한 번 생성과 두 번 생성 (E12 감지·겹밖 문턱).  사실 군집 부트스트랩."""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import grp, fid, mid, NF, OUT, e7_rows
_, _, F7 = e7_rows(); R = np.array([int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]); n = len(R); ism = mid[R]; isf = ~ism
meta = {}
for l in open(f"{OUT}/e7/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
first = np.array([meta[int(i)]["new_type"] == "correct" for i in R]); typ = np.array([meta[int(i)]["new_type"] for i in R]); g = grp[R]; fids = fid[R]
def load(pat, cond=None):
    E = {}
    for f in sorted(glob.glob(f"{OUT}/{pat}")):
        if "_pilot" in f: continue
        for l in open(f, encoding="utf-8"):
            try: d = json.loads(l)
            except Exception: continue
            if cond is None or d["cond"] == cond: E[d["row"]] = d["new_type"] == "correct"
    return E
E3 = load("results_e33_*.jsonl"); EB = load("results_e12_[A-Z].jsonl", "Brel c1"); assert len(E3) >= n and len(EB) >= n, (len(E3), len(EB))
shot = np.array([E3[int(i)] for i in R]); brel = np.array([EB[int(i)] for i in R]); SC = np.load(f"{OUT}/e12_sc.npy")
mi = np.where(ism)[0]; uf = np.unique(fids[mi]); fidx = {f: mi[fids[mi] == f] for f in uf}; rng = np.random.default_rng(0); B = 2000
BF = [np.concatenate([fidx[uf[j]] for j in rng.integers(0, len(uf), len(uf))]) for _ in range(B)]
def cci(d):   # d: 행별 값 (n,), 1~4/5 행 평균의 사실 군집 부트스트랩
    bs = np.array([d[ix].mean() for ix in BF]) * 100; return [round(float(d[mi].mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
def block(p):
    w = ism & ~first; c = ism & first
    return {"1~4/5 이득 [군집 CI]": cci(p.astype(float) - first.astype(float)), "오답 복구 %": round(100 * float(p[w].mean()), 1), "정답 깨짐 %": round(100 * float((~p)[c].mean()), 1),
            "붕괴 복구 %": round(100 * float(p[w & (typ == "broken")].mean()), 1), "entity 복구 %": round(100 * float(p[w & (typ == "entity")].mean()), 1), "5/5 %": round(100 * float(p[isf].mean()), 2)}
def gate(redo):
    sel = np.zeros(n, bool)
    for gg in range(NF):
        tr, te = g != gg, g == gg; bq, bv = 0.0, -1e9
        for q in np.linspace(0, 1, 51):
            th = np.quantile(SC[tr], 1 - q) if q > 0 else np.inf; v = np.where(SC >= th, redo, first)[tr & ism].mean()
            if v > bv + 1e-12: bv, bq = v, q
        th = np.quantile(SC[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = SC[te] >= th
    return sel
O = {"한 번 생성 3-shot": block(shot), "한 번 생성 Brel c1": block(brel)}
s3 = gate(shot); sb = gate(brel); p3 = np.where(s3, shot, first); pb = np.where(sb, brel, first)
O["두 번 생성 3-shot"] = {**block(p3), "다시생성 1~4/5 %": round(100 * float(s3[ism].mean()), 1), "다시생성 5/5 %": round(100 * float(s3[isf].mean()), 1)}
O["두 번 생성 Brel c1"] = {**block(pb), "다시생성 1~4/5 %": round(100 * float(sb[ism].mean()), 1), "다시생성 5/5 %": round(100 * float(sb[isf].mean()), 1)}
O["두 번 생성 차이 Brel - 3shot [군집 CI]"] = cci(pb.astype(float) - p3.astype(float)); O["한 번 생성 차이 Brel - 3shot [군집 CI]"] = cci(brel.astype(float) - shot.astype(float))
d = O["두 번 생성 차이 Brel - 3shot [군집 CI]"]
O["H33.1"] = {"판정": ("조향이 프롬프트보다 낫다" if d[1] > 0 else ("few-shot 이 낫다" if d[2] < 0 else "차이 없음")), "3shot 이 Brel 보다 작다": bool(d[0] > 0)}
print(json.dumps(O, ensure_ascii=False, indent=1)); json.dump(O, open(f"{OUT}/e33_summary.json", "w"), ensure_ascii=False, indent=1)
