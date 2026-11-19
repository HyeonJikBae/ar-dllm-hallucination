"""v11, v13 layout 결과의 순서 일치 / 불일치 분해 (PREREG_poison.md v15 '추가 분해', 탐색).  L 조건.  usage: python explore_order_split.py   출력: results/explore_order_split.json"""
import glob, json
import numpy as np
from grade_layout import load_data, load_pred, s_ok, cube, slope8, boot, XX, RES
from grade_fmt import ratio_boot, verdict
from gen_fmt import select

data = load_data("layout"); PB = {m: load_pred(m, "layout", "base") for m in ("qwen", "dream")}
Q0 = [q for q in data if all(s_ok(PB[m], data, q) for m in PB)]; sub = {r["qid"] for r in select(list(data.values()), 2)}
def split(Y, Q):
    n = len(Q); con, inc, xc, xi = np.zeros((n, 5)), np.zeros((n, 5)), np.zeros((n, 5)), np.zeros((n, 5))
    for i, q in enumerate(Q):
        rc = 1 if data[q]["lb_slot"] == 0 else 0
        con[i], inc[i], xc[i], xi[i] = Y[i, :, rc], Y[i, :, 1 - rc], XX[:, rc], XX[:, 1 - rc]
    return con, inc, xc, xi
def s8(y, x): return 8 * (y * x).sum(1) / (x ** 2).sum(1)
def row(P, Q):
    Y = cube(P, data, Q, "L"); c, i, xc, xi = split(Y, Q)
    return dict(n=len(Q), 전체=round(100 * Y.mean(), 1), 일치=round(100 * c.mean(), 1), 불일치=round(100 * i.mean(), 1), 차=boot(i.mean(1) - c.mean(1)), 기울기x8_전체=boot(slope8(Y)), 기울기x8_일치=boot(s8(c, xc)), 기울기x8_불일치=boot(s8(i, xi)))
R = {}
print("[Qwen 형식별, v10 주 집합 중 그 형식 S 정답]")
for f in ("base", "placebo", "sand", "pre"):
    P = load_pred("qwen", "layout", f); Q = [q for q in Q0 if s_ok(P, data, q)]; R[f"qwen_{f}"] = row(P, Q); print(f" {f:8s}", R[f"qwen_{f}"])
Qs = [q for q in Q0 if q in sub]
print("\n[Dream, 200 문항 안, 완료 문항]")
for n, p in [("base", PB["dream"]), ("sand", load_pred("dream", "layout", "sand")), ("placebo", load_pred("dream", "layout", "placebo"))] + [(a, load_pred("dream", "layout", a)) for a in ("l2r_m16", "ent_s4", "ent_s1", "l2r_m8", "l2r_m64")]:
    Q = [q for q in Qs if all((q, c) in p for c in data[q]["ctx"] if c[0] in "LS")]
    if len(Q) < len(Qs): print(f" {n:8s} 진행 중 ({len(Q)}/{len(Qs)} 문항) 건너뜀"); continue
    Q = [q for q in Q if s_ok(p, data, q)]; R[f"dream_{n}"] = row(p, Q); print(f" {n:8s}", R[f"dream_{n}"])
print("\n[격차 잠정치: placebo 를 뺀 집합.  최종 판정은 grade_fmt.py]")
Ps = {m: load_pred(m, "layout", "sand") for m in PB}; Qg = [q for q in Qs if all((q, c) in Ps["dream"] for c in data[q]["ctx"] if c[0] in "LS") and all(s_ok(Ps[m], data, q) for m in PB)]
it = lambda P, m: cube(P, data, Qg, "L").mean((1, 2)); Gb = it(PB["qwen"], 0) - it(PB["dream"], 0); Gs = it(Ps["qwen"], 0) - it(Ps["dream"], 0)
r = ratio_boot(Gb, Gs); R["격차_잠정"] = dict(n=len(Qg), G_base=boot(Gb), G_sand=boot(Gs), base_minus_sand=boot(Gb - Gs), R=r, 판정=verdict(r, boot(Gb - Gs)), qwen_sand_minus_base=boot(it(Ps["qwen"], 0) - it(PB["qwen"], 0)), dream_sand_minus_base=boot(it(Ps["dream"], 0) - it(PB["dream"], 0))); print(R["격차_잠정"])
json.dump(R, open(f"{RES}/explore_order_split.json", "w"), ensure_ascii=False, indent=1)
