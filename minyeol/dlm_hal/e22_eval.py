"""E22 평가 (PREREG.md H22.1, H22.2, CPU).  report_e6_e8_e9.py 와 같은 가중.  usage: python followup_mk/e22_eval.py"""
import collections, json
import numpy as np
O = "followup_mk"
R = collections.defaultdict(lambda: collections.defaultdict(dict))
for l in open(f"{O}/results_e22.jsonl", encoding="utf-8"):
    d = json.loads(l); R[d["cond"]][d["set"]][d["row"]] = d["new_type"]
E9 = collections.defaultdict(dict)
for l in open(f"{O}/results_e9.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["cond"] == "base": E9[d["set"]][d["row"]] = d["new_type"]
WS = {"broken": 310, "entity": 2749, "correct": 2875}; MID = list(WS)
rate = lambda c, s, t="correct": 100 * float(np.mean([x == t for x in R[c][s].values()]))
def wacc(c): return sum(WS[s] * rate(c, s) for s in MID) / sum(WS.values())
rng = np.random.default_rng(0); base_rows = {s: sorted(R["base"][s]) for s in R["base"]}
def gain_ci(c, B=4000):
    """1~4/5 가중 정답률 이득 (c - base), 행 재표본 (집합별)."""
    cor = {s: np.array([R[c][s][i] == "correct" for i in base_rows[s]], float) for s in MID}; bas = {s: np.array([R["base"][s][i] == "correct" for i in base_rows[s]], float) for s in MID}
    tot = sum(WS.values()); g = []
    for _ in range(B):
        v = 0.0
        for s in MID:
            ix = rng.integers(0, len(base_rows[s]), len(base_rows[s])); v += WS[s] * (cor[s][ix].mean() - bas[s][ix].mean())
        g.append(100 * v / tot)
    return [round(wacc(c) - wacc("base"), 2), round(float(np.percentile(g, 2.5)), 2), round(float(np.percentile(g, 97.5)), 2)]
out = {}
same = sum(R["base"][s][i] == E9[s][i] for s in R["base"] for i in R["base"][s] if i in E9[s]); tot = sum(1 for s in R["base"] for i in R["base"][s] if i in E9[s])
out["sanity_base_대_E9_base_유형일치"] = f"{same}/{tot}"
for c in R:
    r = dict(entity=round(rate(c, "entity"), 1), 정답행=round(rate(c, "correct"), 1), 붕괴행의_정답=round(rate(c, "broken"), 1), 붕괴행의_붕괴=round(rate(c, "broken", "broken"), 1),
             five=round(rate(c, "five"), 1), zero=round(rate(c, "zero"), 1), mid가중=round(wacc(c), 2))
    if c != "base": r["이득[CI]"] = gain_ci(c)
    out[c] = r
bs = out["base"]; ar = out["ar"]
out["판정"] = {"H22.1": bool(ar["붕괴행의_붕괴"] <= 50 and ar["five"] >= 95), "H22.2": bool(ar["이득[CI]"][0] >= 3)}
json.dump(out, open(f"{O}/e22_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
