"""프롬프트 형식 비교 채점 (PREREG_poison.md v11).  usage: python grade_fmt.py [--natural]     출력: results/summary_fmt.json"""
import glob, json, os, sys
import numpy as np
from grade import f1
from grade_layout import load_data, load_pred, L, s_ok, cube, slope8, boot, holm, M, RES, HERE

FMTS = ("base", "sand", "placebo", "pre")


def ratio_boot(g0, g1, seed=0, nb=10000):
    """감소율 R = 1 - mean(g1) / mean(g0) 와 문항 bootstrap CI."""
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(g0), (nb, len(g0))); a, b = g0[ix].mean(1), g1[ix].mean(1)
    r = 1 - b / np.where(a == 0, np.nan, a)
    return [round(100 * float(1 - g1.mean() / g0.mean()), 1), round(100 * float(np.nanpercentile(r, 2.5)), 1), round(100 * float(np.nanpercentile(r, 97.5)), 1)]


def verdict(R, ci):
    if ci[1] <= 0 or R[0] < 25: return "줄지 않음"
    return "크게 줄어듦" if R[0] >= 50 else "부분"


def layout():
    data = load_data("layout"); ok = lambda P, q: s_ok(P, data, q)
    Pq = {f: load_pred("qwen", "layout", f) for f in FMTS}; Pd = {f: load_pred("dream", "layout", f) for f in ("base", "sand", "placebo")}
    S = dict(S_acc=dict(qwen={f: round(100 * np.mean([L(Pq[f], data, q, c) == "정답" for q in data for c in ("S_e", "S_l")]), 1) for f in FMTS},
                        dream={f: round(100 * np.mean([L(Pd[f], data, q, c) == "정답" for q in data if (q, "S_e") in Pd[f] for c in ("S_e", "S_l")]), 1) for f in Pd}))
    print("S 정답률:", S["S_acc"])
    Q0 = [q for q in data if ok(Pq["base"], q) and ok(Pd["base"], q)]; Qq = [q for q in Q0 if ok(Pq["sand"], q) and ok(Pq["placebo"], q)]
    sub = [q for q in Q0 if all((q, c) in Pd[f] for f in ("sand", "placebo") for c in data[q]["ctx"] if c[0] in "LS")]
    Qg = [q for q in sub if all(ok(P[f], q) for P in (Pq, Pd) for f in ("sand", "placebo"))]; S.update(n_v10=len(Q0), n_qwen=len(Qq), n_sub=len(sub), n_gap=len(Qg))
    print(f"v10 주 집합 {len(Q0)}, Qwen 형식 비교 {len(Qq)}, 격차 비교 {len(Qg)} (200 문항 안 {len(sub)})")
    item = lambda A: A.mean((1, 2))
    for c in ("U", "L"):
        Y = {f: cube(Pq[f], data, Qq, c).astype(float) for f in FMTS}; E = {f: cube(Pq[f], data, Qq, c, "그 밖").astype(float) for f in FMTS}; d = {}
        d["형식별"] = {f: dict(다른값=round(100 * Y[f].mean(), 1), 그밖=round(100 * E[f].mean(), 1), 기울기x8=boot(slope8(Y[f]))) for f in FMTS}
        q = lambda a, b: item(Y[a]) - item(Y[b])
        d["H11.1"] = dict(sand_minus_placebo=boot(q("sand", "placebo")), sand_minus_base=boot(q("sand", "base")), placebo_minus_base=boot(q("placebo", "base")), pre_minus_base=boot(q("pre", "base")))
        d["H11.2_slope_sand_minus_placebo"] = boot(slope8(Y["sand"]) - slope8(Y["placebo"])); d["slope_sand_minus_base"] = boot(slope8(Y["sand"]) - slope8(Y["base"]))
        S[c] = d; print(f"\n[{c} Qwen]"); [print("  ", k, v) for k, v in d.items()]
    S["holm_H11_1"] = dict(zip(("U", "L"), holm([S[c]["H11.1"]["sand_minus_placebo"][3] for c in ("U", "L")]))); print("Holm H11.1", S["holm_H11_1"])
    YQ = {f: cube(Pq[f], data, Qg, "L").astype(float) for f in ("base", "sand", "placebo")}; YD = {f: cube(Pd[f], data, Qg, "L").astype(float) for f in ("base", "sand", "placebo")}
    G = {f: item(YQ[f]) - item(YD[f]) for f in YQ}
    d = dict(Dream={f: dict(다른값=round(100 * YD[f].mean(), 1), 기울기x8=boot(slope8(YD[f]))) for f in YD}, Qwen={f: dict(다른값=round(100 * YQ[f].mean(), 1), 기울기x8=boot(slope8(YQ[f]))) for f in YQ},
             G={f: boot(G[f]) for f in G}, G_base_minus_sand=boot(G["base"] - G["sand"]), R=ratio_boot(G["base"], G["sand"]), dream_sand_minus_base=boot(item(YD["sand"]) - item(YD["base"])))
    d["판정"] = verdict(d["R"], d["G_base_minus_sand"]); d["질문접근_해석가능"] = bool(S["L"]["H11.1"]["sand_minus_placebo"][2] < 0); S["격차_L"] = d; print("\n[격차 L]"); [print("  ", k, v) for k, v in d.items()]
    json.dump(S, open(f"{RES}/summary_fmt_layout.json", "w"), ensure_ascii=False, indent=1)


def natural():
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_anchor.jsonl")}
    base = {"qwen": "qwen_anchorbs1", "dream": "dream_anchorbs1_s16"}
    def load(path): return {(r["qid"], r["cond"]): r["pred"] for r in map(json.loads, open(f"{RES}/{path}.jsonl"))}
    def load2(n): return load(n) if os.path.exists(f"{RES}/{n}.jsonl") else {(r["qid"], r["cond"]): r["pred"] for p in sorted(glob.glob(f"{RES}/{n}.sh*.jsonl")) for r in map(json.loads, open(p))}
    P = {m: {"base": load(base[m]), "sand": load2(f"{m}_anchor_sand"), "placebo": load2(f"{m}_anchor_placebo")} for m in M}
    Q = [q for q in data if all((q, c) in P[m][f] for m in M for f in P[m] for c in ("gold", "hard3"))]; S = dict(n=len(Q))
    F = lambda m, f, c: np.array([max(f1(P[m][f][(q, c)], a) for a in data[q]["answers"]) for q in Q])
    D = {(m, f): F(m, f, "gold") - F(m, f, "hard3") for m in M for f in P[m]}
    S["F1"] = {m: {f: dict(gold=round(100 * F(m, f, "gold").mean(), 1), hard3=round(100 * F(m, f, "hard3").mean(), 1)) for f in P[m]} for m in M}
    S["피해"] = {m: {f: boot(D[m, f]) for f in P[m]} for m in M}
    S["Qwen_sand_minus_placebo"] = boot(D["qwen", "sand"] - D["qwen", "placebo"]); S["Qwen_sand_minus_base"] = boot(D["qwen", "sand"] - D["qwen", "base"])
    S["hard3_F1_Qwen_sand_minus_placebo"] = boot(F("qwen", "sand", "hard3") - F("qwen", "placebo", "hard3"))
    G = {f: D["qwen", f] - D["dream", f] for f in ("base", "sand", "placebo")}
    S["G"] = {f: boot(G[f]) for f in G}; S["G_base_minus_sand"] = boot(G["base"] - G["sand"]); S["R"] = ratio_boot(G["base"], G["sand"]); S["판정"] = verdict(S["R"], S["G_base_minus_sand"])
    print(json.dumps(S, ensure_ascii=False, indent=1)); json.dump(S, open(f"{RES}/summary_fmt_natural.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    natural() if "--natural" in sys.argv else layout()
