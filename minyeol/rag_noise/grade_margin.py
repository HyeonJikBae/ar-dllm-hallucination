"""연속 척도 순서 효과 채점 (PREREG_poison.md v16).  usage: python grade_margin.py [--gate-only]   출력: results/summary_margin.json"""
import json, sys
import numpy as np
from grade_layout import load_data, load_pred, L, s_ok, boot, M, RES
from grade_order import ALIAS


def main(gate_only=False):
    data = load_data("order"); P = {m: load_pred(m, "order", "base") for m in M}
    G = {m: {(r["qid"], r["cond"]): r for r in map(json.loads, open(f"{RES}/margin_order_{m}.jsonl"))} for m in M}
    done = {q for q in data if all((q, c) in G[m] for m in M for c in data[q]["ctx"])}
    Q0 = [q for q in data if q in done and all(s_ok(P[m], data, q) for m in M)]
    Q = [q for q in Q0 if all(G[m][(q, c)]["a"] != G[m][(q, c)]["o"] for m in M for c in data[q]["ctx"])]
    conds = [c for c in data[next(iter(data))]["meta"] if data[next(iter(data))]["meta"][c].get("kind") in ("T", "C")]
    mg = lambda m, q, c: G[m][(q, c)]["la"] - G[m][(q, c)]["lo"]
    print(f"문항 완료 {len(done)}, 주 집합 {len(Q0)}, 첫 토큰이 다른 문항 {len(Q)}")
    R = dict(n=len(Q), n_main=len(Q0)); gate = {}
    for m in M:
        n = k = t = 0
        for q in Q:
            for c in conds:
                l = L(P[m], data, q, c)
                if l in ("정답", "공격"): n += 1; k += (mg(m, q, c) > 0) == (l == "정답")
                t += G[m][(q, c)]["top"] in (G[m][(q, c)]["a"], G[m][(q, c)]["o"])
        gate[m] = dict(일치=round(100 * k / n, 1), n=n, top_후보=round(100 * t / (len(Q) * len(conds)), 1)); print(f"  G16 {m}: m>0 와 정답 일치 {gate[m]['일치']}% (n={n}), argmax 가 두 후보 중 하나 {gate[m]['top_후보']}%")
    R["G16"] = gate; ok = all(gate[m]["일치"] >= 90 for m in M); R["G16_통과"] = ok; print("  게이트 통과" if ok else "  게이트 미달: 비교 무효")
    if gate_only: return
    meta = lambda q, c: data[q]["meta"][c]; tpl = np.array([data[q]["template"] for q in Q]); ts = sorted(set(tpl))
    def item(m, pred):
        return np.array([np.mean([mg(m, q, c) for c in conds if pred(meta(q, c), c)]) for q in Q])
    def cluster(v, nb=10000):
        mm = np.array([v[tpl == t].mean() for t in ts]); b = mm[np.random.default_rng(1).integers(0, len(ts), (nb, len(ts)))].mean(1); return [round(float(mm.mean()), 2), round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)]
    def ci(a): return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    idx = np.random.default_rng(0).integers(0, len(Q), (10000, len(Q)))
    def section(name, base):
        o = {m: item(m, lambda d, c: base(d, c) and d["cong"]) for m in M}; i = {m: item(m, lambda d, c: base(d, c) and not d["cong"]) for m in M}; out = {}
        Db = {}; Sb = {}
        for m in M:
            D = o[m] - i[m]; Db[m] = D[idx].mean(1); Sb[m] = Db[m] / o[m][idx].mean(1); s = D.mean() / o[m].mean()
            out[m] = dict(m_일치=round(float(o[m].mean()), 2), m_불일치=round(float(i[m].mean()), 2), D=[round(float(D.mean()), 2), *[round(x, 2) for x in ci(Db[m])]], D_틀=cluster(D), 몫=[round(float(s), 3), *ci(Sb[m])])
            print(f"  {name} {m}: m 일치 {out[m]['m_일치']}  불일치 {out[m]['m_불일치']}  D {out[m]['D']}  틀 {out[m]['D_틀']}  몫 {out[m]['몫']}")
        q, d = M.index("qwen"), M.index("dream"); out["r_D"] = [round(float(np.mean(o["dream"] - i["dream"]) / np.mean(o["qwen"] - i["qwen"])), 3), *ci(Db["dream"] / Db["qwen"])]; out["r_s"] = [round(float(out["dream"]["몫"][0] / out["qwen"]["몫"][0]), 3), *ci(Sb["dream"] / Sb["qwen"])]
        print(f"  {name} r_D {out['r_D']}  r_s {out['r_s']}"); R[name] = out; return out
    isT = lambda d, c: d["kind"] == "T"; isC = lambda d, c: d["kind"] == "C" or c in ALIAS.values()   # grade_order 와 같은 C (별칭 포함)
    print("\n[H16 T x>0]"); h = section("T_x>0", lambda d, c: isT(d, c) and d["x"] > 0)
    print("\n[보조]"); section("T_x<0", lambda d, c: isT(d, c) and d["x"] < 0); section("T_전체", isT); section("C", isC)
    rd, rs = h["r_D"], h["r_s"]
    if not ok: v = "무효"
    elif rd[1] >= .5: v = "순서 효과 크기는 비슷하고 Dream 은 여유가 크다"
    elif rd[2] < .5 and rs[2] < .5: v = "Dream 의 순서 의존은 실제로 작다"
    elif rd[2] < .5 and rs[1] >= .5: v = "비례 축소"
    else: v = "판단 보류"
    R["판정"] = v; print("\n판정:", v)
    print("\n[회귀 m ~ 불일치 + x + xB (T + C), 탐색]"); R["회귀"] = {}
    for m in M:
        X = np.array([[1, not meta(q, c)["cong"], meta(q, c)["x"], meta(q, c)["xB"]] for q in Q for c in conds], float); Y = np.array([mg(m, q, c) for q in Q for c in conds]); nc = len(conds)
        b = np.linalg.lstsq(X, Y, rcond=None)[0]; bs = []
        for s in idx[:2000]:
            ii = np.concatenate([np.arange(j * nc, (j + 1) * nc) for j in s]); bs.append(np.linalg.lstsq(X[ii], Y[ii], rcond=None)[0])
        bs = np.array(bs); R["회귀"][m] = dict(절편=round(float(b[0]), 2), 불일치=[round(float(b[1]), 2), *[round(float(z), 2) for z in np.percentile(bs[:, 1], [2.5, 97.5])]], x=[round(float(b[2]), 2), *[round(float(z), 2) for z in np.percentile(bs[:, 2], [2.5, 97.5])]], xB=[round(float(b[3]), 2), *[round(float(z), 2) for z in np.percentile(bs[:, 3], [2.5, 97.5])]], 불일치_대_x=[round(float(b[1] / b[2]), 2) if b[2] else None, *[round(float(z), 2) for z in np.percentile(bs[:, 1] / bs[:, 2], [2.5, 97.5])]])
        print(f"  {m}: {R['회귀'][m]}")
    print("\n[C 6 칸의 m, 탐색]"); R["C칸"] = {}
    for x in (-4, 0, 4):
        for xb in (-4, 4):
            R["C칸"][f"{x},{xb}"] = {m: round(float(item(m, lambda d, c: isC(d, c) and d["x"] == x and d["xB"] == xb).mean()), 2) for m in M}; print(f"  x {x:+d} xB {xb:+d}: {R['C칸'][f'{x},{xb}']}")
    R["S_m"] = {m: round(float(np.mean([mg(m, q, c) for q in Q for c in ("S_e", "S_l")])), 2) for m in M}; print("\nS 조건 평균 m:", R["S_m"])
    json.dump(R, open(f"{RES}/summary_margin.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main("--gate-only" in sys.argv)
