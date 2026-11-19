"""E26 (PREREG.md, CPU): 첫 칸 중심 감지.  usage: python followup_mk/e26_first_slot.py"""
import collections, glob, json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import grp, k, fid, mid, NF, OUT, MODEL, REV, extract_answer, e7_rows
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); done = np.load(f"{D7}/done.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]; n = len(RR)
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); LP = np.asarray(np.load(f"{D7}/lp.npy", mmap_mode="r")[J]).astype(np.float32)
MG = np.asarray(np.load(f"{D7}/mg.npy", mmap_mode="r")[J]).astype(np.float32); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J])
kk = k[RR]; g7 = grp[RR]; fid7 = fid[RR]; corr = np.array([meta[int(i)]["new_type"] == "correct" for i in RR]); typ = np.array([meta[int(i)]["new_type"] for i in RR])
isz, isf, ism = kk == 0, kk == 5, (kk >= 1) & (kk <= 4); mw = ism & ~corr
W = np.where(isz, 44110 / isz.sum(), np.where(isf, 5165 / isf.sum(), 1.0))
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1)
S = {}
S["S1 -(step0 첫칸 margin)"] = -MG[:, 0, 0]
S["S2 첫칸 확정 step"] = lastm[:, 0].astype(float)
m0 = MK[:, :, 0]; a0 = IDS[:, :, 0]
flips = np.zeros(n); dist = np.zeros(n)
for j in range(n):
    st_ = np.where(m0[j])[0]; seq = a0[j, st_]
    flips[j] = (seq[1:] != seq[:-1]).sum() if len(seq) > 1 else 0; dist[j] = len(set(seq.tolist()))
S["S3 첫칸 argmax 바뀐 횟수"] = flips; S["S4 첫칸 후보 종류 수"] = dist
lastc = np.clip(lastm[:, 0], 0, None); S["S5 -(확정 순간 logprob)"] = -LP[np.arange(n), lastc, 0]; S["S6 -(step0 첫칸 logprob)"] = -LP[:, 0, 0]
# 궤적 신호 SC (e11 과 같은 계산, 전 행)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SPs = set(tok.all_special_ids)
committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
def dc(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SPs), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
ANS = [[dc(r) for r in np.where(MK[:, s], IDS[:, s], committed)] for s in range(32)]
SC = np.array([np.mean([ANS[s][j] != ANS[31][j] for s in range(31)]) for j in range(n)]); S0 = {"SC 궤적 (최종답과 다른 step 비율)": SC}
rk = lambda x: (np.argsort(np.argsort(x)) + 0.5) / len(x)
C = {"C1 SC+S2": rk(SC) + rk(S["S2 첫칸 확정 step"]), "C2 SC+S1": rk(SC) + rk(S["S1 -(step0 첫칸 margin)"]), "C3 SC+S3": rk(SC) + rk(S["S3 첫칸 argmax 바뀐 횟수"]),
     "C4 SC+S1+S2+S3": rk(SC) + rk(S["S1 -(step0 첫칸 margin)"]) + rk(S["S2 첫칸 확정 step"]) + rk(S["S3 첫칸 argmax 바뀐 횟수"])}
ALL = {**S0, **S, **C}
mi = np.where(ism)[0]
def within(s):
    g = collections.defaultdict(list)
    for j in mi: g[fid7[j]].append(j)
    a = [roc_auc_score(~corr[v], s[v]) for v in g.values() if 0 < corr[v].sum() < len(v)]; return float(np.mean(a))
O = {"표본": dict(n=n, mid=int(ism.sum()), 오답=int(mw.sum()), 붕괴=int((mw & (typ == "broken")).sum()), entity=int((mw & (typ == "entity")).sum()))}; det = {}
for nm, s in ALL.items():
    pos_b = mw & (typ == "broken"); pos_e = mw & (typ == "entity")
    det[nm] = {"T2 가중": round(float(roc_auc_score(mw, s, sample_weight=W)), 3), "T3 안": round(float(roc_auc_score(~corr[mi], s[mi])), 3), "T3 같은사실": round(within(s), 3),
               "오답 대 5/5": round(float(roc_auc_score(np.r_[np.ones(mw.sum()), np.zeros(isf.sum())], np.r_[s[mw], s[isf]])), 3),
               "붕괴 대 정답(1~4/5)": round(float(roc_auc_score(np.r_[np.ones(pos_b.sum()), np.zeros((ism & corr).sum())], np.r_[s[pos_b], s[ism & corr]])), 3),
               "entity 대 정답(1~4/5)": round(float(roc_auc_score(np.r_[np.ones(pos_e.sum()), np.zeros((ism & corr).sum())], np.r_[s[pos_e], s[ism & corr]])), 3)}
    print(nm, det[nm], flush=True)
O["감지"] = det
# ---- 두 번 생성 이득 (E12 Brel c1 다시 생성; R = 1~4/5 + 5/5 2,000)
_, _, F7 = e7_rows(); R = np.array([int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]); pos = {int(i): j for j, i in enumerate(RR)}; ix = np.array([pos[int(i)] for i in R]); nR = len(R)
E = {}
for f in sorted(glob.glob(f"{OUT}/results_e12_[A-Z].jsonl")):
    for l in open(f, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        if d["cond"] == "Brel c1": E[d["row"]] = d["new_type"] == "correct"
redo = np.array([E[int(i)] for i in R]); first = corr[ix]; ismR = ism[ix]; isfR = isf[ix]; rng = np.random.default_rng(0)
def gain(sc, q):
    sel = sc >= np.quantile(sc, 1 - q); p = np.where(sel, redo, first); return (p.astype(float) - first.astype(float))[ismR], p, sel
tp = {}
for nm, s in ALL.items():
    sR = s[ix]; o = {}
    for q in (.3, .4, .5):
        d_, p, sel = gain(sR, q); bs = np.array([d_[rng.integers(0, len(d_), len(d_))].mean() for _ in range(2000)]) * 100
        o[f"상위{int(q * 100)}%"] = {"이득": round(float(d_.mean() * 100), 2), "CI": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)], "5/5": round(100 * float(p[isfR].mean()), 2), "다시생성 5/5 %": round(100 * float(sel[isfR].mean()), 1)}
    tp[nm] = o; print(nm, json.dumps(o, ensure_ascii=False), flush=True)
O["두번생성"] = tp
dif = {}
for nm in C:
    for q in (.3, .4, .5):
        a, _, _ = gain(C[nm][ix], q); b, _, _ = gain(SC[ix], q); dd = a - b; bs = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]) * 100
        dif[f"{nm} 상위{int(q * 100)}% - SC"] = [round(float(dd.mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
O["조합 - SC 이득 차이"] = dif; print(json.dumps(dif, ensure_ascii=False))
O["판정"] = {"H26.1": bool(any(det[c]["T3 같은사실"] >= .75 for c in C)), "H26.2": bool(any(v[0] >= 1 and v[1] > 0 for v in dif.values()))}
print("판정", O["판정"]); json.dump(O, open(f"{OUT}/e26_summary.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DONE")
