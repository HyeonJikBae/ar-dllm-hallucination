"""E27 평가 (PREREG.md H27.1, H27.2, CPU).  usage: python followup_mk/e27_eval.py [--pilot]"""
import collections, json, sys
import numpy as np
PIL = "--pilot" in sys.argv; O = "followup_mk"; sfx = "_pilot" if PIL else ""; out = {}
# ---------- E27a
d = json.load(open(f"{O}/results_e27a{sfx}.json")); rows = d["rows"]; out["sanity_t0입력_칸0argmax_E7일치"] = d["sanity_base_top1_E7일치"]
W = [r for r in rows if r["kind"] == "wrong"]; Cn = [r for r in rows if r["kind"] == "control"]
Wc = [r for r in W if r["n_cand"] > 0]; Cc = [r for r in Cn if r["n_cand"] > 0]
A = sum(1 for r in W if (not r["base_is_gold"]) and r["flips_gold"]); Cnt = sum(1 for r in W if r["flips_any"]); Bnt = sum(1 for r in Cn if r["flips_any"])
out["E27a"] = {"오답 행": len(W), "정답 대조 행": len(Cn), "오답 중 확정된 다른 칸이 있는 행": len(Wc), "정답 중 확정된 다른 칸이 있는 행": len(Cc),
               "오답 중 t0 입력에서 이미 칸 0 top-1 이 공여 정답 토큰인 행 %": round(100 * float(np.mean([r["base_is_gold"] for r in W])), 1),
               "A: 단일 칸을 가리면 칸 0 top-1 이 정답 토큰이 되는 오답 행 %": round(100 * A / len(W), 1), "C: 단일 칸을 가리면 top-1 이 바뀌는 오답 행 %": round(100 * Cnt / len(W), 1),
               "B: 단일 칸을 가리면 top-1 이 바뀌는 정답 대조 행 %": round(100 * Bnt / len(Cn), 1), "A/C": round(A / Cnt, 3) if Cnt else None,
               "(확정 칸 있는 행만 분모) A %": round(100 * A / len(Wc), 1) if Wc else None, "(확정 칸 있는 행만) C %": round(100 * Cnt / len(Wc), 1) if Wc else None, "(확정 칸 있는 행만) B %": round(100 * Bnt / len(Cc), 1) if Cc else None}
# 위치 분포: (행, 칸) 쌍당 비율
def bucket(c): return "1" if c == 1 else "2-3" if c <= 3 else "4-8" if c <= 8 else "9-16" if c <= 16 else "17-31"
bk = ["1", "2-3", "4-8", "9-16", "17-31"]; risk = collections.Counter(); fg = collections.Counter(); fa = collections.Counter(); fb = collections.Counter(); riskb = collections.Counter()
for r in W:
    for c in r["cand"]: risk[bucket(c)] += 1
    for c in r["flips_gold"]: fg[bucket(c)] += 1
    for c in r["flips_any"]: fa[bucket(c)] += 1
for r in Cn:
    for c in r["cand"]: riskb[bucket(c)] += 1
    for c in r["flips_any"]: fb[bucket(c)] += 1
out["E27a 위치 (칸 범위: 쌍 수 / 오답 정답으로 바뀜 % / 오답 아무로나 바뀜 % / 정답 대조 바뀜 %)"] = {b: (risk[b], round(100 * fg[b] / max(risk[b], 1), 2), round(100 * fa[b] / max(risk[b], 1), 2), round(100 * fb[b] / max(riskb[b], 1), 2)) for b in bk}
sp = collections.Counter(); spn = collections.Counter()
for r in W:
    for c, s_ in zip(r["cand"], r["cand_special"]): spn[s_] += 1; sp[s_] += int(c in r["flips_any"])
out["E27a 오답: 가린 칸이 특수 토큰인가 (False=내용 칸)별 top-1 변화 %"] = {str(k_): (spn[k_], round(100 * sp[k_] / max(spn[k_], 1), 2)) for k_ in spn}
rk = lambda rs, key: {"top1 %": round(100 * float(np.mean([r[key] == 0 for r in rs])), 1), "top2 안 %": round(100 * float(np.mean([r[key] <= 1 for r in rs])), 1), "top5 안 %": round(100 * float(np.mean([r[key] <= 4 for r in rs])), 1), "중앙 순위": float(np.median([r[key] for r in rs]))}
out["E27a 공여 정답 토큰 순위: t0 입력 (오답)"] = rk(W, "base_rank"); out["E27a 순위: 확정 칸 전부 가림 (step 0 상태, 오답)"] = rk(W, "all_rank")
out["E27a 대조(정답 행) 자기 토큰 순위: t0 입력"] = rk(Cn, "base_rank"); out["E27a 대조 순위: 전부 가림"] = rk(Cn, "all_rank")
out["E27a logp 변화 (전부 가림 - t0 입력) 평균: 오답 / 정답 대조"] = [round(float(np.mean([r["all_logp"] - r["base_logp"] for r in W])), 3), round(float(np.mean([r["all_logp"] - r["base_logp"] for r in Cn])), 3)]
dl_w = [x for r in W for x in r["dlogp"]]; dl_c = [x for r in Cn for x in r["dlogp"]]
out["E27a 단일 칸 가림의 logp 변화 평균 / 상위 5% 분위: 오답, 정답 대조"] = {"오답": [round(float(np.mean(dl_w)), 3), round(float(np.percentile(dl_w, 95)), 3)], "정답 대조": [round(float(np.mean(dl_c)), 3), round(float(np.percentile(dl_c, 95)), 3)]}
a_pct = 100 * A / len(W); ac = (A / Cnt) if Cnt else 0
out["H27.1"] = {"A %": round(a_pct, 1), "A/C": round(ac, 3), "판정": bool(a_pct >= 15 and ac >= 0.3), "B 와 비교 (C%, B%)": [out["E27a"]["C: 단일 칸을 가리면 top-1 이 바뀌는 오답 행 %"], out["E27a"]["B: 단일 칸을 가리면 top-1 이 바뀌는 정답 대조 행 %"]]}
# ---------- E27b
R = collections.defaultdict(lambda: collections.defaultdict(dict)); extra = collections.defaultdict(list)
for l in open(f"{O}/results_e27b{sfx}.jsonl", encoding="utf-8"):
    x = json.loads(l); R[x["cond"]][x["set"]][x["row"]] = x["new_type"]
    if x["cond"] == "칸0먼저": extra[x["set"]].append((x["row"], x["step0_tok_is_E7_tok"], x["step0_tok_special"]))
E9 = collections.defaultdict(dict)
for l in open(f"{O}/results_e9.jsonl", encoding="utf-8"):
    x = json.loads(l)
    if x["cond"] == "base": E9[x["set"]][x["row"]] = x["new_type"]
WS = {"broken": 310, "entity": 2749, "correct": 2875}; MID = list(WS)
rate = lambda c, s, t="correct": 100 * float(np.mean([v == t for v in R[c][s].values()]))
wacc = lambda c: sum(WS[s] * rate(c, s) for s in MID) / sum(WS.values())
rng = np.random.default_rng(0); base_rows = {s: sorted(R["base"][s]) for s in R["base"]}
def gain_ci(c, B=4000):
    cor = {s: np.array([R[c][s][i] == "correct" for i in base_rows[s]], float) for s in MID}; bas = {s: np.array([R["base"][s][i] == "correct" for i in base_rows[s]], float) for s in MID}; g = []
    for _ in range(B):
        v = 0.0
        for s in MID: ix = rng.integers(0, len(base_rows[s]), len(base_rows[s])); v += WS[s] * (cor[s][ix].mean() - bas[s][ix].mean())
        g.append(100 * v / sum(WS.values()))
    return [round(wacc(c) - wacc("base"), 2), round(float(np.percentile(g, 2.5)), 2), round(float(np.percentile(g, 97.5)), 2)]
same = sum(R["base"][s][i] == E9[s][i] for s in R["base"] for i in R["base"][s] if i in E9[s]); tot = sum(1 for s in R["base"] for i in R["base"][s] if i in E9[s])
b = {"sanity_base_대_E9_base_유형일치": f"{same}/{tot}"}
for c in ("base", "칸0먼저"):
    r = dict(붕괴행의_정답=round(rate(c, "broken"), 1), entity행의_정답=round(rate(c, "entity"), 1), 정답행_유지=round(rate(c, "correct"), 1), five=round(rate(c, "five"), 1), zero=round(rate(c, "zero"), 1), mid가중=round(wacc(c), 2))
    if c != "base": r["이득[CI]"] = gain_ci(c)
    b[c] = r
b["step0 칸0 argmax 가 E7 첫 생성 칸0 토큰과 같은 비율 % (집합별)"] = {s: round(100 * float(np.mean([e[1] for e in v])), 1) for s, v in extra.items()}
b["step0 칸0 argmax 가 특수 토큰인 비율 %"] = {s: round(100 * float(np.mean([e[2] for e in v])), 1) for s, v in extra.items()}
out["E27b"] = b; out["H27.2"] = {"이득[CI]": b["칸0먼저"]["이득[CI]"], "판정": bool(b["칸0먼저"]["이득[CI]"][1] > 0)}
json.dump(out, open(f"{O}/e27_summary{sfx}.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False, indent=1))
