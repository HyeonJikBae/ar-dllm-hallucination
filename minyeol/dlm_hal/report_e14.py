"""E14 집계: 1~4/5 가중 정답률, 같은 실행 base 대비 이득 부트스트랩 CI, 고침/깨짐, 5/5, 0/5, 다시 가림 발동, E9 base 와 일치."""
import collections, json, sys
import numpy as np
O = "followup_mk"; PIL = "--pilot" in sys.argv
E = collections.defaultdict(dict); SET = {}; NR = collections.defaultdict(dict); ND = collections.defaultdict(dict)
for l in open(f"{O}/results_e14{'_pilot' if PIL else ''}.jsonl"):
    d = json.loads(l); E[d["cond"]][d["row"]] = d["new_type"] == "correct"; SET[d["row"]] = d["set"]; NR[d["cond"]][d["row"]] = d["n_remask"]; ND[d["cond"]][d["row"]] = d["n_defer_steps"]
e9 = {}
for l in open(f"{O}/results_e9.jsonl"):
    d = json.loads(l)
    if d["cond"] == "base": e9[d["row"]] = d["new_type"] == "correct"
R = sorted(E["base"]); sets = np.array([SET[i] for i in R]); WS = {"broken": 310, "entity": 2749, "correct": 2875}
cnt = {s_: max(1, (sets == s_).sum()) for s_ in WS}; ws = np.array([WS[s_] / cnt[s_] if s_ in WS else 0. for s_ in sets]); mS = np.isin(sets, list(WS))
base = np.array([E["base"][i] for i in R])
wacc = lambda x, msk, w=None: float(100 * (x * (ws if w is None else w))[msk].sum() / (ws if w is None else w)[msk].sum())
rng = np.random.default_rng(0); POS = {s_: np.where(sets == s_)[0] for s_ in WS}; BIX = {s_: rng.integers(0, len(POS[s_]), (2000, len(POS[s_]))) for s_ in WS}
out = {"E9base_일치": round(100 * float(np.mean([E["base"][i] == e9[i] for i in R if i in e9])), 2), "조건": {}}
for c in E:
    x = np.array([E[c][i] for i in R]); d = []
    for b in range(2000):
        ix = np.concatenate([POS[s_][BIX[s_][b]] for s_ in WS]); o_ = np.ones(len(ix), bool); d.append(wacc(x[ix], o_, ws[ix]) - wacc(base[ix], o_, ws[ix]))
    tw = ws[mS].sum(); fix = (~base & x & mS); brk = (base & ~x & mS)
    nr = np.array([NR[c][i] for i in R]); nd = np.array([ND[c][i] for i in R])
    out["조건"][c] = {"1~4/5": round(wacc(x, mS), 2), "이득[CI]": [round(wacc(x, mS) - wacc(base, mS), 2), round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
                      "고침": round(100 * float(ws[fix].sum() / tw), 1), "깨짐": round(100 * float(ws[brk].sum() / tw), 1),
                      **{s_: round(100 * float(x[sets == s_].mean()), 1) for s_ in ("broken", "entity", "correct", "zero", "five")},
                      "다시가림_발동행%": round(100 * float((nr > 0).mean()), 1), "다시가림_평균": round(float(nr.mean()), 2), "미룸_발동행%": round(100 * float((nd > 0).mean()), 1)}
v = {c: (o["이득[CI]"][0] > 0 and o["five"] >= 98) for c, o in out["조건"].items() if c != "base"}
out["H14_2"] = {"통과조건": [c for c, ok in v.items() if ok and c.startswith("T2M")], "하한>0": [c for c, o in out["조건"].items() if c != "base" and o["이득[CI]"][1] > 0]}
json.dump(out, open(f"{O}/report_e14{'_pilot' if PIL else ''}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
