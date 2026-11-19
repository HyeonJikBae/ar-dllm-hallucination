"""순서 대 거리 채점 (PREREG_poison.md v15).  usage: python grade_order.py [order]   출력: results/summary_order.json"""
import json, sys
import numpy as np
from grade_layout import load_data, load_pred, L, s_ok, boot, holm, M, RES

ALIAS = {"C1b0": "T4_10", "C1b0_sw": "T4_10_sw", "C7b8": "T4_78", "C7b8_sw": "T4_78_sw"}
def verdict(b):
    if b[1] >= -5 and b[2] <= 5: return "순서 효과 없음"
    return "거리와 별개의 순서 효과" if b[1] > 0 else ("근접 쪽" if b[2] < 0 else "판단 보류")


def main(tag="order", P=None):
    data = load_data(tag); P = P or {m: load_pred(m, tag, "base") for m in M}
    keys = [q for q in data if all((q, c) in P[m] for m in M for c in data[q]["ctx"])]
    S = dict(n_items=len(data), n_done=len(keys), S_acc={m: round(100 * np.mean([L(P[m], data, q, c) == "정답" for q in keys for c in ("S_e", "S_l")]), 1) for m in M}); Q = [q for q in keys if all(s_ok(P[m], data, q) for m in M)]; S["n_main"] = len(Q)
    print(f"{tag}: 문항 {len(data)}, 완료 {len(keys)}, 주 집합 {len(Q)}, S {S['S_acc']}")
    tpl = np.array([data[q]["template"] for q in Q]); fam = np.array([data[q].get("family", "old") for q in Q]); ts = sorted(set(tpl))
    def cluster(v, seed=1, nb=10000):
        v = np.asarray(v, float); mm = np.array([v[tpl == t].mean() for t in ts]); b = mm[np.random.default_rng(seed).integers(0, len(ts), (nb, len(ts)))].mean(1)
        return [round(100 * mm.mean(), 1), round(100 * np.percentile(b, 2.5), 1), round(100 * np.percentile(b, 97.5), 1)]
    C_ALL = ["C1b8", "C1b8_sw", "C4b0", "C4b0_sw", "C4b8", "C4b8_sw", "C7b0", "C7b0_sw"]; T = [c for c in data[Q[0]]["meta"] if data[Q[0]]["meta"][c].get("kind") == "T"]; C = C_ALL + list(ALIAS)
    def meta(q, c): return data[q]["meta"][ALIAS.get(c, c)]
    def y(m, q, c, label="공격"): return L(P[m], data, q, ALIAS.get(c, c)) == label
    # cong, x, g, side 는 x_first 에 따라 조건 이름마다 문항별로 다르므로 문항별 집합을 쓴다
    def by_item(m, pred, label="공격"):
        out = []
        for q in Q:
            s = [c for c in T + C if pred(meta(q, c), c)]; out.append(np.mean([y(m, q, c, label) for c in s]))
        return np.array(out)
    def order_eff(m, base):
        i = by_item(m, lambda d, c: base(d, c) and not d["cong"]); o = by_item(m, lambda d, c: base(d, c) and d["cong"]); return i - o, o, i
    isT = lambda d, c: c in T; isC = lambda d, c: c in C
    R = {}
    def report(name, base):
        R[name] = {}
        for m in M:
            d, o, i = order_eff(m, base); R[name][m] = dict(일치=round(100 * o.mean(), 1), 불일치=round(100 * i.mean(), 1), 차=boot(d), 차_틀=cluster(d))
            print(f"  {name} {m}: 일치 {R[name][m]['일치']}  불일치 {R[name][m]['불일치']}  차 {R[name][m]['차']}  틀 {R[name][m]['차_틀']}")
        a = order_eff("qwen", base)[0]; b = order_eff("dream", base)[0]; R[name]["qwen_minus_dream"] = boot(a - b); print(f"  {name} Qwen-Dream {R[name]['qwen_minus_dream']}")
    print("\n[H15.1 교차 배치 C]"); report("C", isC); report("C_k4", lambda d, c: isC(d, c) and d["a"] == 4)
    h = holm([R["C"][m]["차"][3] for m in M]); R["C"]["holm"] = dict(zip(M, h)); print("  Holm", R["C"]["holm"])
    print("\n[H15.2 같은 쪽 T]"); report("T_x>0", lambda d, c: isT(d, c) and d["x"] > 0); report("T_x<0", lambda d, c: isT(d, c) and d["x"] < 0); report("T_전체", isT)
    h = holm([R["T_x>0"][m]["차"][3] for m in M]); R["T_x>0"]["holm"] = dict(zip(M, h)); print("  Holm", R["T_x>0"]["holm"])
    for m in M: R["T_x>0"][m]["판정"] = verdict(R["T_x>0"][m]["차"]); print(f"  {m} 판정: {R['T_x>0'][m]['판정']}")
    print("\n[H15.4 g x 쪽 x x 부호 x 순서]"); R["표"] = {}
    for m in M:
        print(f" {m}")
        for side in ("after", "before"):
            for sgn in (1, -1):
                line = []
                for g in (1, 2, 4):
                    for cong in (True, False):
                        v = by_item(m, lambda d, c: isT(d, c) and d["side"] == side and d["g"] == g and np.sign(d["x"]) == sgn and d["cong"] == cong).mean(); R["표"][f"{m}|{side}|{'+' if sgn > 0 else '-'}|g{g}|{'일치' if cong else '불일치'}"] = round(100 * v, 1); line.append(f"g{g} {'일치' if cong else '불치'} {100 * v:5.1f}")
                print(f"   {side:6s} x{'>' if sgn > 0 else '<'}0: " + "  ".join(line))
        for side in ("after", "before"):
            g4 = by_item(m, lambda d, c: isT(d, c) and d["side"] == side and d["g"] == 4 and d["x"] > 0 and d["cong"]); g1 = by_item(m, lambda d, c: isT(d, c) and d["side"] == side and d["g"] == 1 and d["x"] > 0 and d["cong"])
            R[f"g4-g1_{m}_{side}"] = boot(g4 - g1); print(f"   x>0 일치 g4-g1 {side}: {R[f'g4-g1_{m}_{side}']}")
    print("\n[H15.5 C 일치 칸의 거리]"); R["C일치_거리"] = {}
    for m in M:
        d = by_item(m, lambda dd, c: isC(dd, c) and dd["cong"] and dd["x"] == 4) - by_item(m, lambda dd, c: isC(dd, c) and dd["cong"] and dd["x"] == -4); R["C일치_거리"][m] = boot(d); print(f"  {m}: {R['C일치_거리'][m]}")
    q5, d5 = R["C일치_거리"]["qwen"], R["C일치_거리"]["dream"]; R["H15.5"] = bool(d5[1] >= -5 and d5[2] <= 5 and q5[1] > 5); print("  '순서가 맞을 때 거리 효과는 Qwen 에만 있다':", R["H15.5"])
    print("\n[틀 묶음, T x>0 불일치-일치]")
    for m in M:
        d = order_eff(m, lambda dd, c: isT(dd, c) and dd["x"] > 0)[0]; R.setdefault("틀묶음", {})[m] = {f: boot(d[fam == f]) for f in sorted(set(fam))}; by = [d[tpl == t].mean() for t in ts]; R["틀묶음"][m]["양수틀"] = int(sum(v > 0 for v in by)); print(f"  {m}: {R['틀묶음'][m]}")
    print("\n[그 밖 비율]"); R["그밖"] = {m: {k: round(100 * by_item(m, lambda d, c: f(d, c), "그 밖").mean(), 1) for k, f in (("T", isT), ("C", isC))} for m in M}; print(" ", R["그밖"])
    S.update(R); json.dump(S, open(f"{RES}/summary_{tag}.json", "w"), ensure_ascii=False, indent=1); return S


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "order")
