"""v10 탐색 분석 (사전 등록 아님).  칸별 Qwen-Dream 격차, B 연결 칸별 분해, 틀 단위 bootstrap, 원문 예시.  usage: python explore_layout.py   출력: results/explore_layout.json"""
import json, random
import numpy as np
from grade_layout import load_data, load_pred, L, s_ok, cube, slope8, boot, KS, M, RES

data = load_data("layout"); P = {m: load_pred(m, "layout", "base") for m in M}
Q = [q for q in data if all((q, c) in P[m] for m in M for c in data[q]["ctx"]) and all(s_ok(P[m], data, q) for m in M)]
tpl = np.array([data[q]["template"] for q in Q]); fam = np.array([data[q].get("family", "old") for q in Q])
Y = {(m, c): cube(P[m], data, Q, c) for m in M for c in ("U", "L")}
R = {"n": len(Q)}


def cluster(v, seed=1, nb=10000):
    """틀 단위 bootstrap.  틀 20개를 복원추출하고 각 틀의 문항 평균을 쓴다."""
    v = np.asarray(v, float); ts = sorted(set(tpl)); m = np.array([v[tpl == t].mean() for t in ts]); rng = np.random.default_rng(seed)
    b = m[rng.integers(0, len(ts), (nb, len(ts)))].mean(1)
    return [round(100 * m.mean(), 1), round(100 * np.percentile(b, 2.5), 1), round(100 * np.percentile(b, 97.5), 1)]


print("n", len(Q)); print("\n[1] 칸별 Qwen - Dream 격차 (문항 쌍 bootstrap).  역할 0 = 다른 사실 뒤 칸 6, 1 = 앞 칸 2")
R["격차"] = {}
for c in ("U", "L"):
    print(f" {c}")
    for j, k in enumerate(KS):
        for r, name in enumerate(("뒤칸", "앞칸")):
            d = Y["qwen", c][:, j, r] - Y["dream", c][:, j, r]; x = (k - 2 - (6 - k)) * (1 if r == 0 else -1) if 1 < k < 7 else (-4 if k == 1 else 4) * (1 if r == 0 else -1)
            b = boot(d); R["격차"][f"{c}{k}_{name}"] = dict(x=x, qwen=round(100 * Y["qwen", c][:, j, r].mean(), 1), dream=round(100 * Y["dream", c][:, j, r].mean(), 1), gap=b)
            print(f"   k={k} {name} x={x:+d}  Qwen {100 * Y['qwen', c][:, j, r].mean():5.1f}  Dream {100 * Y['dream', c][:, j, r].mean():5.1f}  차 {b[0]:6.1f} [{b[1]:6.1f},{b[2]:6.1f}]")

print("\n[2] 격차의 구성: 전체 평균 격차 중 각 칸의 몫 (L, U)")
for c in ("U", "L"):
    g = {f"k{k}_{n}": (Y["qwen", c][:, j, r] - Y["dream", c][:, j, r]).mean() for j, k in enumerate(KS) for r, n in enumerate(("뒤칸", "앞칸"))}; tot = sum(g.values())
    print(f" {c} 평균 격차 {100 * tot / 10:.1f}%p; 각 칸 몫(%):", {k: round(100 * v / tot, 1) for k, v in g.items()})
    R[f"격차몫_{c}"] = {k: round(100 * v / tot, 1) for k, v in g.items()}

print("\n[3] 틀 단위 bootstrap (틀 20개)")
R["틀단위"] = {}
for c in ("U", "L"):
    s = {m: slope8(Y[m, c]) for m in M}
    t = dict(qwen=cluster(s["qwen"]), dream=cluster(s["dream"]), gap=cluster(s["qwen"] - s["dream"]),
             qwen_k1=cluster(Y["qwen", c][:, 0, 0] - Y["qwen", c][:, 0, 1]), dream_k1=cluster(Y["dream", c][:, 0, 0] - Y["dream", c][:, 0, 1]),
             qwen_role=cluster(Y["qwen", c][:, :, 0].mean(1) - Y["qwen", c][:, :, 1].mean(1)), dream_role=cluster(Y["dream", c][:, :, 0].mean(1) - Y["dream", c][:, :, 1].mean(1)))
    R["틀단위"][c] = t; print(f" {c}", t)

print("\n[4] 틀별 기울기x8 범위와 칸 1 앞칸 격차")
R["틀별"] = {}
for c in ("U", "L"):
    for m in M:
        s = slope8(Y[m, c]); by = sorted((round(100 * s[tpl == t].mean(), 1), int(t)) for t in set(tpl))
        print(f" {c} {m}: 최저 {by[:3]} 최고 {by[-3:]}"); R["틀별"][f"{c}_{m}"] = by

lb = np.array([data[q]["ctx"]["L1"].index("LB") for q in Q]); R["B칸"] = {"n0": int((lb == 0).sum()), "n8": int((lb == 8).sum())}
print(f"\n[5] L 조건, B 연결 칸별 다른 값 선택 (B칸 0: {(lb == 0).sum()}문항, B칸 8: {(lb == 8).sum()}문항)")
for m in M:
    print(f" {m}")
    for j, k in enumerate(KS):
        for r, n in enumerate(("뒤칸", "앞칸")):
            a, b_ = Y[m, "L"][lb == 0, j, r], Y[m, "L"][lb == 8, j, r]
            R["B칸"][f"{m}_k{k}_{n}"] = [round(100 * a.mean(), 1), round(100 * b_.mean(), 1)]
            print(f"   k={k} {n}: B칸0 {100 * a.mean():5.1f}  B칸8 {100 * b_.mean():5.1f}")
for m in M:
    s = slope8(Y[m, "L"]); print(f" {m} 기울기x8 B칸0 {boot(s[lb == 0])}  B칸8 {boot(s[lb == 8])}")
    R["B칸"][f"{m}_slope"] = [boot(s[lb == 0]), boot(s[lb == 8])]
    d = Y[m, "L"][:, :, 0].mean(1) - Y[m, "L"][:, :, 1].mean(1); print(f"    역할효과 B칸0 {boot(d[lb == 0])}  B칸8 {boot(d[lb == 8])}"); R["B칸"][f"{m}_role"] = [boot(d[lb == 0]), boot(d[lb == 8])]

print("\n[6] Dream 의 L 조건 오류가 다른 사실-B 연결의 거리와 관계있는가: d(B, 다른 사실) 2 vs 6")
for m in M:
    # 역할 0 = 다른 사실 칸 6: B칸 8 이면 2칸, B칸 0 이면 6칸.  역할 1 = 다른 사실 칸 2: B칸 0 이면 2칸, B칸 8 이면 6칸.
    near = np.concatenate([Y[m, "L"][lb == 8, :, 0].mean(1), Y[m, "L"][lb == 0, :, 1].mean(1)]); far = np.concatenate([Y[m, "L"][lb == 0, :, 0].mean(1), Y[m, "L"][lb == 8, :, 1].mean(1)])
    print(f" {m}: 2칸 {100 * near.mean():.1f}  6칸 {100 * far.mean():.1f}"); R.setdefault("B거리", {})[m] = [round(100 * near.mean(), 1), round(100 * far.mean(), 1)]

print("\n[7] 원문 예시: Dream L1 앞칸에서 다른 값 (Qwen 은 정답) / Qwen L7 뒤칸에서 다른 값 (Dream 은 정답)")
random.seed(0)
def cond_for(q, k, c, role):
    return next(cc for cc in (f"{c}{k}", f"{c}{k}_sw") if data[q]["meta"][cc]["oth_slot"] == (6 if role == 0 else 2))
for (k, role, bad, good) in ((1, 1, "dream", "qwen"), (7, 0, "qwen", "dream")):
    rows = [q for q in Q if L(P[bad], data, q, cond_for(q, k, "L", role)) == "공격" and L(P[good], data, q, cond_for(q, k, "L", role)) == "정답"]
    print(f"  k={k} {bad} 만 틀림: {len(rows)}건")
    for q in random.sample(rows, min(4, len(rows))):
        cc = cond_for(q, k, "L", role); ids = data[q]["ctx"][cc]
        print(f"   {q} {cc}: Q={data[q]['question']} | 정답={data[q]['ans'][cc]} 다른값={data[q]['other'][cc]} | qwen={P['qwen'][q, cc]!r} dream={P['dream'][q, cc]!r} | 칸={ids}")
        print("      LA:", data[q]["docs"][ids[k]]["text"][:110])
json.dump(R, open(f"{RES}/explore_layout.json", "w"), ensure_ascii=False, indent=1)

# ---- 순서 일치 분석 (탐색).  연결 순서 = 사실 순서 (첫 연결 <-> 첫 사실) 이면 일치.
# B 칸 0: A 는 둘째 연결 -> 둘째 사실(칸 6) 이 정답이면 일치 = 다른 사실이 앞 칸 (역할 1).  B 칸 8: A 는 첫 연결 -> 다른 사실이 뒤 칸 (역할 0) 이면 일치.
print("\n[8] 순서 일치 / 불일치 (L).  이 설계에서는 d(B, 다른 사실) = 2 / 6 과 완전히 겹친다")
from grade_layout import XX
def split(Yc):
    n = len(Q); con = np.zeros((n, 5)); inc = np.zeros((n, 5)); xc = np.zeros((n, 5)); xi = np.zeros((n, 5))
    for i in range(n):
        rc = 1 if lb[i] == 0 else 0; ri = 1 - rc
        con[i] = Yc[i, :, rc]; inc[i] = Yc[i, :, ri]; xc[i] = XX[:, rc]; xi[i] = XX[:, ri]
    return con, inc, xc, xi
def s8(y, x): return 8 * (y * x).sum(1) / (x ** 2).sum(1)
R["순서"] = {}
for m in M:
    con, inc, xc, xi = split(Y[m, "L"]); d = {}
    d["일치"] = round(100 * con.mean(), 1); d["불일치"] = round(100 * inc.mean(), 1); d["불일치-일치"] = boot(inc.mean(1) - con.mean(1)); d["불일치-일치_틀"] = cluster(inc.mean(1) - con.mean(1))
    d["기울기x8_일치"] = boot(s8(con, xc)); d["기울기x8_불일치"] = boot(s8(inc, xi))
    d["x별_일치"] = {int(x): round(100 * con[xc == x].mean(), 1) for x in (-4, -2, 0, 2, 4)}; d["x별_불일치"] = {int(x): round(100 * inc[xi == x].mean(), 1) for x in (-4, -2, 0, 2, 4)}
    d["k4"] = dict(일치=round(100 * con[:, 2].mean(), 1), 불일치=round(100 * inc[:, 2].mean(), 1), 차=boot(inc[:, 2] - con[:, 2]))
    R["순서"][m] = d; print(f" {m}:", json.dumps(d, ensure_ascii=False))
cq, iq, xc, xi = split(Y["qwen", "L"]); cd, id_, _, _ = split(Y["dream", "L"])
R["순서"]["격차"] = dict(일치=boot(cq.mean(1) - cd.mean(1)), 불일치=boot(iq.mean(1) - id_.mean(1)), 기울기x8_일치=boot(s8(cq, xc) - s8(cd, xc)), 기울기x8_불일치=boot(s8(iq, xi) - s8(id_, xi)))
print(" Qwen-Dream:", R["순서"]["격차"])
print("\n[9] 음성 대조: U 에는 B 가 없으므로 B칸 표시로 나눠도 달라지면 안 된다")
for m in M:
    con, inc, _, _ = split(Y[m, "U"]); R["순서"][f"{m}_U"] = dict(일치표시=round(100 * con.mean(), 1), 불일치표시=round(100 * inc.mean(), 1), 차=boot(inc.mean(1) - con.mean(1))); print(f" {m}:", R["순서"][f"{m}_U"])
    print("    B칸0:", round(100 * Y[m, "U"][lb == 0].mean(), 1), " B칸8:", round(100 * Y[m, "U"][lb == 8].mean(), 1), " 틀별 불일치-일치 양수:", sum((inc.mean(1) - con.mean(1))[tpl == t].mean() > 0 for t in set(tpl)), "/ 20")
con, inc, _, _ = split(Y["dream", "L"]); by = [(inc.mean(1) - con.mean(1))[tpl == t].mean() for t in sorted(set(tpl))]; print(" L 틀별 불일치-일치 양수 (dream):", sum(v > 0 for v in by), "/ 20")
con, inc, _, _ = split(Y["qwen", "L"]); by = [(inc.mean(1) - con.mean(1))[tpl == t].mean() for t in sorted(set(tpl))]; print(" L 틀별 불일치-일치 양수 (qwen):", sum(v > 0 for v in by), "/ 20")
for f in ("old", "new"):
    for m in M:
        con, inc, _, _ = split(Y[m, "L"]); print(f" L {f} {m} 불일치-일치:", boot((inc.mean(1) - con.mean(1))[fam == f]))
json.dump(R, open(f"{RES}/explore_layout.json", "w"), ensure_ascii=False, indent=1)
