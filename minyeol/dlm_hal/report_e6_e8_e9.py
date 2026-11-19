"""E6, E8, E9 집계 (PREREG.md 판정표)."""
import collections, json, os
import numpy as np
O = "followup_mk"; out = {}
def load(f):
    R = collections.defaultdict(lambda: collections.defaultdict(dict))
    for l in open(f): d = json.loads(l); R[d["cond"]][d["set"]][d["row"]] = d["new_type"]
    return R
def rate(R, c, s, t="correct"): v = list(R[c][s].values()); return round(100 * np.mean([x == t for x in v]), 1) if v else None
WS = {"broken": 310, "entity": 2749, "correct": 2875}
def wmid(R, c): return round(sum(WS[s] * rate(R, c, s) for s in WS) / sum(WS.values()), 1)
if os.path.exists(f"{O}/results_e6.jsonl"):
    R = load(f"{O}/results_e6.jsonl"); t = {c: {s: (rate(R, c, s), rate(R, c, s, "broken")) for s in R[c]} for c in R}
    g = max(rate(R, "G0 c1", "broken"), rate(R, "G1 c1", "broken")); kc = rate(R, "K c1.0", "broken")
    H1 = kc >= 15 and kc >= g + 8; H2 = rate(R, "B⊥K c1", "broken") >= .8 * rate(R, "B c1", "broken"); H3 = rate(R, "K c1.0", "zero") <= 3
    verdict = {(True, True): "아는 상태와 답하는 상태가 따로 기여", (True, False): "B 효과의 상당 부분이 아는 상태 축", (False, True): "고침은 지식이 아니라 답하는 상태", (False, False): "혼재"}[(H1, H2)]
    out["E6"] = dict(조건별_정답_붕괴=t, H6_1=H1, H6_2=H2, H6_3=H3, 판정=verdict)
for name in ("e8", "e9"):
    f = f"{O}/results_{name}.jsonl"
    if not os.path.exists(f): continue
    R = load(f); t = {c: dict(**{s: rate(R, c, s) for s in R[c]}, 붕괴행의붕괴=rate(R, c, "broken", "broken"), mid가중=wmid(R, c)) for c in R}; out[name.upper()] = dict(조건별=t)
    b = t["base"]
    if name == "e8":
        ok = [c for c in t if c != "base" and t[c]["붕괴행의붕괴"] <= 50 and t[c]["five"] >= 95]; out["E8"]["H8_1"] = dict(통과조건=ok, 판정="붕괴는 디코딩 순서·예산에 민감" if ok else "디코딩 조건으로 붕괴가 크게 줄지 않음")
    else:
        ace = [c for c in t if c.startswith("ACE") and t[c]["mid가중"] - b["mid가중"] >= 3 and t[c]["five"] >= 97]
        fix = lambda c: np.mean([R[c]["broken"][i] == "correct" and R["base"]["broken"][i] != "correct" for i in R["base"]["broken"]])
        out["E9"].update(H9_1=dict(통과=ace), H9_2=dict(늦게시작_고침=round(100 * fix("B c1 s8-31"), 1), 전구간_고침=round(100 * fix("B c1"), 1), 통과=bool(fix("B c1 s8-31") >= .7 * fix("B c1"))),
                         H9_3=dict(관계별=t["Brel c1"]["broken"], B=t["B c1"]["broken"], 통과=bool(t["Brel c1"]["broken"] - t["B c1"]["broken"] >= 5)))
json.dump(out, open(f"{O}/report_e6_e8_e9.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for kx, v in out.items(): print("==", kx); print(json.dumps(v, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:4000])
