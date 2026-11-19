"""E30 (PREREG.md, CPU): 모집단 기준 2 단계 감지.  1 단계 = 가운데K L27 s0 (사실 수준), 2 단계 = SC (궤적).  다시 생성 = E12 Brel c1.  0/5 는 정확도 변화 0, 비용만."""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import rows, grp, k, mid, unit, NF, OUT, MODEL, REV, extract_answer
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); done = np.load(f"{D7}/done.npy"); meta = {}; cfg = json.load(open(f"{D7}/config.json"))
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]; n = len(RR)
kk = k[RR]; g7 = grp[RR]; corr = np.array([meta[int(i)]["new_type"] == "correct" for i in RR]); isz, isf, ism = kk == 0, kk == 5, (kk >= 1) & (kk <= 4); mw = ism & ~corr
W = np.where(isz, 44110 / isz.sum(), np.where(isf, 5165 / isf.sum(), 1.0)); rel = np.array([rows[int(i)]["relation"] for i in RR])
# 1 단계: 가운데K L27 s0 (e10b 와 같은 계산)
Xh = np.asarray(np.load(f"{D7}/hid.npy", mmap_mode="r")[J, cfg["steps"].index(0), cfg["layers"].index(27), 0], dtype=np.float32); pk = np.zeros(n); mm = np.zeros(n)
for g in range(NF):
    tr = g7 != g; te = ~tr; ds = [Xh[tr & isf & (rel == r_)].mean(0) - Xh[tr & isz & (rel == r_)].mean(0) for r_ in np.unique(rel) if (tr & isf & (rel == r_)).sum() >= 10 and (tr & isz & (rel == r_)).sum() >= 10]
    K = unit(np.mean(ds, 0)); pk[te] = Xh[te] @ K; mm[te] = np.median(Xh[tr & ism] @ K)
S1 = -np.abs(pk - mm)
# 2 단계: SC
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SPs = set(tok.all_special_ids)
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
def dc(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SPs), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
ANS = [[dc(r) for r in np.where(MK[:, s], IDS[:, s], committed)] for s in range(32)]
SC = np.array([np.mean([ANS[s][j] != ANS[31][j] for s in range(31)]) for j in range(n)]); np.save(f"{OUT}/e30_scores.npy", np.stack([S1, SC]))
def wrank(x):
    o = np.argsort(x); cw = np.cumsum(W[o]); r = np.empty(n); r[o] = (cw - W[o] / 2) / W.sum(); return r
COMB = wrank(S1) * wrank(SC)
# 다시 생성 결과 (E12 Brel c1): 1~4/5, 5/5.  0/5 는 첫 생성과 같다고 둠.
E = {}
for f in sorted(glob.glob(f"{OUT}/results_e12_[A-Z].jsonl")):
    for l in open(f, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        if d["cond"] == "Brel c1": E[d["row"]] = d["new_type"] == "correct"
redo = np.array([E.get(int(i), c) if not z else c for i, c, z in zip(RR, corr, isz)]); assert sum(int(i) in E for i in RR[~isz]) == (~isz).sum()
rng = np.random.default_rng(0); mi = np.where(ism)[0]
def policy(s, b):
    o = np.argsort(-s); cw = np.cumsum(W[o]); sel = np.zeros(n, bool); sel[o[cw <= b * W.sum()]] = True; return sel
def stats(sel):
    p = np.where(sel, redo, corr); d = (p.astype(float) - corr.astype(float))
    return {"1~4/5 이득": round(100 * float(d[ism].mean()), 2), "모집단 정확도 변화": round(100 * float((d * W).sum() / W.sum()), 3), "5/5 정답 %": round(100 * float(p[isf].mean()), 2),
            "걸린 것 중 1~4/5 오답 몫 %": round(100 * float(W[sel & mw].sum() / max(W[sel].sum(), 1e-9)), 1), "1~4/5 오답 재현율 %": round(100 * float(sel[mw].mean()), 1),
            "0/5 걸린 %": round(100 * float(sel[isz].mean()), 1), "5/5 걸린 %": round(100 * float(sel[isf].mean()), 1)}
O = {"기저: 모집단 중 1~4/5 오답 %": round(100 * float(W[mw].sum() / W.sum()), 2), "첫 생성 모집단 정확도 %": round(100 * float((corr * W).sum() / W.sum()), 2)}
from sklearn.metrics import roc_auc_score
for nm, s in (("SC", SC), ("1단계 가운데K", S1), ("결합", COMB)): O[f"T2 가중 AUROC {nm}"] = round(float(roc_auc_score(mw, s, sample_weight=W)), 3)
for b in (.05, .10, .20, .30):
    row = {nm: stats(policy(s, b)) for nm, s in (("SC", SC), ("1단계", S1), ("결합", COMB))}
    a = policy(COMB, b); c = policy(SC, b); dd = (np.where(a, redo, corr).astype(float) - np.where(c, redo, corr).astype(float))[mi]
    bs = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]) * 100
    row["결합-SC 1~4/5 이득 차이 [CI]"] = [round(float(dd.mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
    O[f"예산 {int(b * 100)}%"] = row; print(f"예산 {int(b * 100)}%", json.dumps(row, ensure_ascii=False), flush=True)
r20 = O["예산 20%"]; O["H30.1"] = bool(r20["결합-SC 1~4/5 이득 차이 [CI]"][0] >= 1 and r20["결합"]["걸린 것 중 1~4/5 오답 몫 %"] >= 2 * r20["SC"]["걸린 것 중 1~4/5 오답 몫 %"])
print({k_: v for k_, v in O.items() if not k_.startswith("예산")}); json.dump(O, open(f"{OUT}/e30_summary.json", "w"), ensure_ascii=False, indent=1)
