"""E38 평가 (PREREG.md, CPU).  usage: python followup_mk/e38_eval.py [--pilot]"""
import collections, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import fid, OUT
import glob
PIL = "--pilot" in sys.argv
files = sorted(glob.glob(f"{OUT}/results_e38_pilot*.jsonl")) if PIL else [f for f in sorted(glob.glob(f"{OUT}/results_e38*.jsonl")) if "_pilot" not in f]
D = collections.defaultdict(dict); SET = {}
for fn in files:
    for l in open(fn, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        D[d["cond"]][d["row"]] = d; SET[d["row"]] = d["set"]
O = {"완료": {c: len(v) for c, v in D.items()}}
rows = sorted(D["A"]); sets = np.array([SET[i] for i in rows]); n = len(rows)
POP = {"broken": 310, "entity": 2749, "correct": 2875, "zero": 44110, "five": 5165}
cnt = collections.Counter(sets); ws = np.array([POP[s] / cnt[s] for s in sets]); mS = np.isin(sets, ["broken", "entity", "correct"])
base = np.array([D["A"][i]["new_type"] == "correct" for i in rows])
# sanity: E9 base 와 유형 일치
e9 = {}
for l in open(f"{OUT}/results_e9.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["cond"] == "base": e9[d["row"]] = d["new_type"]
O["sanity_A_대_E9base_유형일치"] = f"{sum(e9.get(i) == D['A'][i]['new_type'] for i in rows)}/{n}"
def wacc(x, m): return float(100 * (x * ws)[m].sum() / ws[m].sum())
# 사실 군집 부트스트랩 (set 층화)
rng = np.random.default_rng(0); NB = 4000; fidr = np.array([fid[i] for i in rows])
strata = {}
for s in set(sets):
    ix = np.where(sets == s)[0]; byf = collections.defaultdict(list)
    for j in ix: byf[fidr[j]].append(j)
    strata[s] = list(byf.values())
draws = []
for b in range(NB):
    sel = []
    for s, groups in strata.items():
        pick = rng.integers(0, len(groups), len(groups))
        for p in pick: sel.extend(groups[p])
    draws.append(np.array(sel))
def ci_gain(x):
    d = [100 * ((x[ix] * ws[ix])[mS[ix]].sum() - (base[ix] * ws[ix])[mS[ix]].sum()) / ws[ix][mS[ix]].sum() for ix in draws]
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]
def fixbrk(x):
    w_ = mS & ~base; c_ = mS & base
    return round(100 * float((ws * (x & w_)).sum() / (ws * w_).sum()), 1), round(100 * float((ws * (~x & c_)).sum() / (ws * c_).sum()), 1)
res = {"A": {"1~4/5": round(wacc(base, mS), 2), "5/5": round(100 * float(base[sets == "five"].mean()), 1), "0/5": round(100 * float(base[sets == "zero"].mean()), 1)}}
for c in [c for c in ["B", "C05", "C1", "D", "E1", "E2", "F", "G"] if c in D and len(D[c]) == n]:
    x = np.array([D[c][i]["new_type"] == "correct" for i in rows]); f_, b_ = fixbrk(x)
    wb = mS & ~base & (sets == "broken"); we = mS & ~base & (sets == "entity")
    res[c] = {"1~4/5": round(wacc(x, mS), 2), "이득": round(wacc(x, mS) - wacc(base, mS), 2), "군집CI": ci_gain(x), "오답복구": f_, "정답깨짐": b_,
              "붕괴복구": round(100 * float(x[wb].mean()), 1) if wb.any() else None, "entity복구": round(100 * float(x[we].mean()), 1) if we.any() else None,
              "5/5": round(100 * float(x[sets == "five"].mean()), 1), "0/5": round(100 * float(x[sets == "zero"].mean()), 1)}
    print(c, json.dumps(res[c], ensure_ascii=False), flush=True)
O["조건별"] = res
# 금 첫 토큰 순위 (칸 0, step 0~3): base 오답인 1~4/5 행과 base 정답인 1~4/5 행
def rk(c, grp_mask, s):
    v = np.array([D[c][i][f"gold_rank_s{s}"] for i in rows]); m_ = grp_mask & (v >= 0)
    if not m_.any(): return None
    vv = v[m_]; return {"n": int(m_.sum()), "top1%": round(100 * float((vv == 0).mean()), 1), "top2%": round(100 * float((vv <= 1).mean()), 1), "중앙순위": float(np.median(vv))}
rank_tab = {}
for c in [c for c in ["A", "B", "C1", "D", "E1", "E2"] if c in D and len(D[c]) == n]:
    rank_tab[c] = {f"오답행 s{s}": rk(c, mS & ~base, s) for s in range(4)}
    rank_tab[c].update({f"정답행 s{s}": rk(c, mS & base, s) for s in (0, 3)})
    # 같은 행 짝 비교: step 0 에서 base 대비 순위가 오른 비율 (오답행)
    if c != "A":
        a0 = np.array([D["A"][i]["gold_rank_s0"] for i in rows]); c0 = np.array([D[c][i]["gold_rank_s0"] for i in rows]); m_ = mS & ~base & (a0 >= 0) & (c0 >= 0)
        rank_tab[c]["오답행 s0 순위 오름/내림 %"] = [round(100 * float((c0[m_] < a0[m_]).mean()), 1), round(100 * float((c0[m_] > a0[m_]).mean()), 1)]
    print("순위", c, json.dumps(rank_tab[c], ensure_ascii=False), flush=True)
O["금토큰순위"] = rank_tab
# 판정
B_ = res.get("B")
h1 = [c for c in ("C05", "C1", "E1", "E2") if c in res and B_ and res[c]["오답복구"] >= 0.9 * B_["오답복구"] and res[c]["정답깨짐"] <= 0.7 * B_["정답깨짐"]]
h2 = (res["C1"]["오답복구"] - res["D"]["오답복구"]) if ("C1" in res and "D" in res) else None
F_ = res.get("F"); G_ = res.get("G")
O["판정"] = {"H38.1": dict(통과=h1, 판정=bool(h1)), "H38.2": dict(C1_minus_D_복구=h2, 판정=bool(h2 is not None and h2 >= 10)),
            "H38.3": dict(F이득=(F_["이득"] if F_ else None), F_CI=(F_["군집CI"] if F_ else None), 판정=bool(F_ and F_["이득"] > 0 and F_["군집CI"][0] > 0),
                          G이득=(G_["이득"] if G_ else None), B이득=(B_["이득"] if B_ else None), G_가_B보다_큼=bool(G_ and B_ and G_["이득"] > B_["이득"]))}
print("판정", json.dumps(O["판정"], ensure_ascii=False), flush=True)
json.dump(O, open(f"{OUT}/e38_summary{'_pilot' if PIL else ''}.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DONE")
