"""거리 용량-반응 채점 (PREREG_poison.md v10).  usage: python grade_layout.py [layout|layout_pilot] [base]   출력: results/summary_<data>_<fmt>.json"""
import glob, json, os, sys
import numpy as np
from grade import norm
from grade_poison_main import lab

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"; KS = (1, 3, 4, 5, 7); M = ("qwen", "dream"); NB = 10000


def load_data(d): return {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_{d}.jsonl")}
def load_pred(m, d, f):
    out = {}
    for p in glob.glob(f"{RES}/{m}_{d}_{f}.jsonl") + glob.glob(f"{RES}/{m}_{d}_{f}.sh*.jsonl"):
        out.update({(r["qid"], r["cond"]): r["pred"] for r in map(json.loads, open(p))})
    return out
def L(P, data, q, c): return lab(P[(q, c)], [data[q]["ans"][c]], data[q]["other"][c])
def s_ok(P, data, q): return all(L(P, data, q, c) == "정답" for c in ("S_e", "S_l"))


def boot(v, seed=0):
    """문항별 값 v 의 평균과 95% CI, 양측 p.  단위 %p."""
    v = np.asarray(v, float); rng = np.random.default_rng(seed); ix = rng.integers(0, len(v), (NB, len(v))); b = v[ix].mean(1)
    p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return [round(100 * v.mean(), 1), round(100 * np.percentile(b, 2.5), 1), round(100 * np.percentile(b, 97.5), 1), round(float(p), 4)]


def cube(P, data, Q, c, label="공격"):
    """[문항, k, 역할].  역할 0 = 다른 사실이 뒤 칸 (6), 1 = 앞 칸 (2)."""
    Y = np.zeros((len(Q), 5, 2))
    for i, q in enumerate(Q):
        for j, k in enumerate(KS):
            for s in ("", "_sw"):
                cc = f"{c}{k}{s}"; role = 0 if data[q]["meta"][cc]["oth_slot"] == 6 else 1
                Y[i, j, role] = L(P, data, q, cc) == label
    return Y


X = np.array([[k - 2 - (6 - k) if k > 2 and k < 6 else (-4 if k == 1 else 4) for _ in range(1)][0] for k in KS])   # 역할 0 의 x: d(A,정답=칸2) - d(A,다른=칸6)
assert X.tolist() == [-4, -2, 0, 2, 4]
XX = np.stack([X, -X], 1)                                                                                       # 역할 1 은 부호 반대


def slope8(Y): return 8 * (Y * XX).sum((1, 2)) / (XX ** 2).sum()          # 문항별 기울기 x 8 = x -4 -> +4 차이
def by_x(Y): return {int(x): Y[:, XX == x].mean(1) for x in (-4, -2, 0, 2, 4)}


def holm(ps):
    o = np.argsort(ps); out = [0.0] * len(ps); cur = 0.0
    for r, i in enumerate(o): cur = max(cur, min(1.0, ps[i] * (len(ps) - r))); out[i] = round(float(cur), 4)
    return out


def analyze(P, data, Q, c):
    Y = cube(P, data, Q, c); bx = by_x(Y); d = {}
    d["n"] = len(Q); d["다른값"] = round(100 * Y.mean(), 1)
    d["그밖"] = round(100 * cube(P, data, Q, c, "그 밖").mean(), 1); d["정답"] = round(100 * cube(P, data, Q, c, "정답").mean(), 1)
    d["x별"] = {int(x): round(100 * v.mean(), 1) for x, v in bx.items()}
    d["칸별"] = {str(k): dict(뒤칸=round(100 * Y[:, j, 0].mean(), 1), 앞칸=round(100 * Y[:, j, 1].mean(), 1)) for j, k in enumerate(KS)}
    d["기울기x8"] = boot(slope8(Y)); d["끝점차"] = boot(bx[4] - bx[-4])
    d["역할효과"] = boot(Y[:, :, 0].mean(1) - Y[:, :, 1].mean(1)); d["역할효과_k4"] = boot(Y[:, 2, 0] - Y[:, 2, 1])
    d["엇갈림_k1"] = boot(Y[:, 0, 0] - Y[:, 0, 1]); d["일치_k7"] = boot(Y[:, 4, 0] - Y[:, 4, 1])
    d["가시성"] = dict(k1_앞칸_연결뒤=round(100 * Y[:, 0, 1].mean(), 1), k3_앞칸_연결앞=round(100 * Y[:, 1, 1].mean(), 1),
                     k5_뒤칸_연결뒤=round(100 * Y[:, 3, 0].mean(), 1), k7_뒤칸_연결앞=round(100 * Y[:, 4, 0].mean(), 1))
    C = cube(P, data, Q, c, "정답")
    d["연결민감"] = round(100 * (C[:, :, 0] * C[:, :, 1]).mean(), 1); d["모두정답"] = round(100 * (C.sum((1, 2)) == 10).mean(), 1)
    st = []
    for q in Q:
        for s in ("", "_sw"): st.append(len({norm(P[(q, f"{c}{k}{s}")]) for k in KS}) == 1)
    d["배치안정"] = round(100 * np.mean(st), 1)
    return d, Y


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "layout"; fmt = sys.argv[2] if len(sys.argv) > 2 else "base"
    data = load_data(tag); P = {m: load_pred(m, tag, fmt) for m in M}
    keys = [q for q in data if all((q, c) in P[m] for m in M for c in data[q]["ctx"])]
    S = dict(n_items=len(data), n_done=len(keys), S_acc={m: round(100 * np.mean([L(P[m], data, q, c) == "정답" for q in keys for c in ("S_e", "S_l")]), 1) for m in M})
    Q = [q for q in keys if all(s_ok(P[m], data, q) for m in M)]; S["n_main"] = len(Q)
    print(f"{tag}/{fmt}: 문항 {len(data)}, 완료 {len(keys)}, 주 집합 {len(Q)}, S 정답률 {S['S_acc']}")
    Ys = {}
    for c in ("U", "L"):
        S[c] = {}
        for m in M:
            S[c][m], Ys[m, c] = analyze(P[m], data, Q, c); d = S[c][m]
            print(f"\n[{c} {m}] 다른값 {d['다른값']}  그밖 {d['그밖']}  정답 {d['정답']}")
            print("   x별:", d["x별"]); print("   칸별:", d["칸별"])
            print("   기울기x8", d["기울기x8"], " 끝점차", d["끝점차"], " 역할효과", d["역할효과"], " k4", d["역할효과_k4"])
            print("   엇갈림(k1)", d["엇갈림_k1"], " 일치(k7)", d["일치_k7"], " 가시성", d["가시성"])
        S[c]["기울기x8_qwen_minus_dream"] = boot(slope8(Ys["qwen", c]) - slope8(Ys["dream", c]))
        print(f"   기울기x8 Qwen - Dream {S[c]['기울기x8_qwen_minus_dream']}")
        fam = np.array([data[q].get("family", "old") for q in Q]); tpl = np.array([data[q]["template"] for q in Q])
        for m in M:
            s8 = slope8(Ys[m, c]); S[c][m]["틀묶음_기울기x8"] = {f: boot(s8[fam == f]) for f in sorted(set(fam))}
            by_t = {int(t): round(100 * s8[tpl == t].mean(), 1) for t in sorted(set(tpl))}; S[c][m]["틀별_기울기x8"] = by_t
            print(f"   {m} 연결민감 {S[c][m]['연결민감']} 배치안정 {S[c][m]['배치안정']} 모두정답 {S[c][m]['모두정답']}  틀 묶음 {S[c][m]['틀묶음_기울기x8']}  양수 틀 {sum(v > 0 for v in by_t.values())}/{len(by_t)}")
    h = holm([S[c]["qwen"]["기울기x8"][3] for c in ("U", "L")]); S["holm_qwen_slope"] = dict(U=h[0], L=h[1]); print("\nHolm (Qwen 기울기):", S["holm_qwen_slope"])
    json.dump(S, open(f"{RES}/summary_{tag}_{fmt}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
