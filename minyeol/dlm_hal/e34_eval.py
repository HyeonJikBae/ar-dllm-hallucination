"""E34 평가 (PREREG.md, CPU): OSCAR 식 다중 체인 감지 대 SC, 가운데K, E30 결합.  행 S, 정답 라벨 = E7 첫 생성 (E9 base 와 100% 일치)."""
import collections, glob, json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import k, fid, OUT, set_S
S = set_S(); rows_S = np.array([i for _, i in S]); sets = np.array([s for s, _ in S])
CH = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{OUT}/results_e34_*.jsonl")):
    if "_pilot" in f: continue
    for l in open(f, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        CH[d["row"]][d["chain"]] = (d["answer"].strip().lower(), d["slots"])
nch = collections.Counter(len(v) for v in CH.values()); print("체인 수 분포", dict(nch), flush=True)
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); done = np.load(f"{D7}/done.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]; pos = {int(r): j for j, r in enumerate(RR)}
S1, SCa = np.load(f"{OUT}/e30_scores.npy"); kk7 = k[RR]; isz7, isf7 = kk7 == 0, kk7 == 5
W7 = np.where(isz7, 44110 / isz7.sum(), np.where(isf7, 5165 / isf7.sum(), 1.0))
def wrank(x):
    o = np.argsort(x); cw = np.cumsum(W7[o]); r = np.empty(len(x)); r[o] = (cw - W7[o] / 2) / W7.sum(); return r
COMB = wrank(S1) * wrank(SCa)
keep = np.array([int(i) in pos and len(CH.get(int(i), {})) == 8 for i in rows_S]); rows_S, sets = rows_S[keep], sets[keep]; n = len(rows_S)
j7 = np.array([pos[int(i)] for i in rows_S]); first = np.array([meta[int(i)]["new_type"] == "correct" for i in rows_S]); kk = k[rows_S]; fids = fid[rows_S]
ism = (kk >= 1) & (kk <= 4); isz = kk == 0; isf = kk == 5; mw = ism & ~first
def ent(xs):
    c = np.array(list(collections.Counter(xs).values()), float); p = c / c.sum(); return float(-(p * np.log(p)).sum())
oscar_ent = np.array([np.mean([ent([CH[int(i)][c][1][s] if len(CH[int(i)][c][1]) > s else -1 for c in range(8)]) for s in range(4)]) for i in rows_S])
oscar_dis = np.array([1 - collections.Counter(CH[int(i)][c][0] for c in range(8)).most_common(1)[0][1] / 8 for i in rows_S])
SC_ = {"OSCAR 칸0~3 엔트로피": oscar_ent, "OSCAR 답 불일치율": oscar_dis, "SC (E7 궤적)": SCa[j7], "가운데K L27 s0": S1[j7], "E30 결합": COMB[j7]}
WS = {"broken": 310 / 310, "entity": 2749 / 300, "correct": 2875 / 300, "zero": 44110 / 300, "five": 5165 / 300}; ws = np.array([WS[s] for s in sets])
mi = np.where(ism)[0]; fg = collections.defaultdict(list)
for j in mi: fg[fids[j]].append(j)
both = [np.array(v) for v in fg.values() if 0 < first[v].sum() < len(v)]
def within(s, groups=both): return float(np.mean([roc_auc_score(~first[v], s[v]) for v in groups]))
rng = np.random.default_rng(0); BG = [[both[t] for t in rng.integers(0, len(both), len(both))] for _ in range(2000)]
O = {"행": int(n), "1~4/5 행": int(ism.sum()), "같은 사실 안 비교 가능한 사실 수": len(both), "점수": {}}
for nm, s in SC_.items():
    O["점수"][nm] = {"T3 같은사실": round(within(s), 3), "T3 1~4/5 안": round(float(roc_auc_score(~first[mi], s[mi])), 3),
                   "1~4/5 오답 대 5/5": round(float(roc_auc_score(np.r_[np.ones(mw.sum()), np.zeros(isf.sum())], np.r_[s[mw], s[isf]])), 3),
                   "1~4/5 오답 대 0/5": round(float(roc_auc_score(np.r_[np.ones(mw.sum()), np.zeros(isz.sum())], np.r_[s[mw], s[isz]])), 3),
                   "T2 모집단 가중": round(float(roc_auc_score(mw, s, sample_weight=ws)), 3)}
    print(nm, O["점수"][nm], flush=True)
best = max(("OSCAR 칸0~3 엔트로피", "OSCAR 답 불일치율"), key=lambda x: O["점수"][x]["T3 같은사실"])
dd = np.array([within(SC_[best], gr) - within(SC_["SC (E7 궤적)"], gr) for gr in BG])
O["OSCAR 최선 - SC (T3 같은사실) [군집 CI]"] = [round(O["점수"][best]["T3 같은사실"] - O["점수"]["SC (E7 궤적)"]["T3 같은사실"], 3), round(float(np.percentile(dd, 2.5)), 3), round(float(np.percentile(dd, 97.5)), 3)]
O["H34.1"] = {"최선 OSCAR 점수": best, "판정": bool(O["OSCAR 최선 - SC (T3 같은사실) [군집 CI]"][0] >= 0.03)}
print(json.dumps({k_: v for k_, v in O.items() if k_ != "점수"}, ensure_ascii=False)); json.dump(O, open(f"{OUT}/e34_summary.json", "w"), ensure_ascii=False, indent=1)
