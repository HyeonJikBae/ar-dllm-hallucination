"""v3 (연결 교환) 데이터에서 연결-사실 순서 일치 교차 확인 (탐색, 새 생성 없음).  usage: python explore_order_v3.py   출력: results/explore_order_v3.json"""
import collections, json
import numpy as np
from grade_layout import boot, RES, HERE
from grade_poison_main import lab

rows = [json.loads(l) for l in open(f"{HERE}/data/musique_bindswap.jsonl")]
P = {m: {(r["qid"], r["cond"]): r["pred"] for r in map(json.loads, open(f"{RES}/{m}_bindswap.jsonl"))} for m in ("qwen", "dream")}
def lb(m, r, c): return lab(P[m][r["qid"], c], [r["ans"][c]], r["other"][c])
Q = [r for r in rows if all(lb(m, r, "S") == "정답" for m in P)]; print("주 집합", len(Q))
def facts(c, r):
    """정답 사실, 다른 사실.  교환이면 A<-Y 이므로 정답 사실 = FY."""
    return ("FY", "FX") if c.endswith("_sw") else ("FX", "FY")
sig = collections.Counter(); out = collections.defaultdict(lambda: collections.defaultdict(list)); per = collections.defaultdict(dict)
for r in Q:
    for c in ("L", "L_sw"):
        ids = r["ctx"][c]; a = next(i for i, d in enumerate(ids) if d in ("LA", "LAs")); b = next(i for i, d in enumerate(ids) if d in ("LB", "LBs"))
        fa, fo = facts(c, r); sa, so = ids.index(fa), ids.index(fo)
        cong = (a < b) == (sa < so); x = abs(a - sa) - abs(a - so); key = (a, b, sa, so)
        sig[key] += 1
        for m in P:
            y = lb(m, r, c) == "공격"; out[m][cong].append(y); out[m]["cong" if cong else "inc", key].append(y); out[m][("x", x, cong)].append(y)
            per[m].setdefault(r["qid"], {})[cong] = y
print("배치 종류", len(sig))
R = {}
for m in P:
    print(f"\n{m}: 일치 {100 * np.mean(out[m][True]):.1f} ({len(out[m][True])})  불일치 {100 * np.mean(out[m][False]):.1f} ({len(out[m][False])})")
    R[m] = dict(일치=round(100 * np.mean(out[m][True]), 1), 불일치=round(100 * np.mean(out[m][False]), 1), 배치={})
    for k in sorted((k for k in out[m] if isinstance(k, tuple) and k[0] in ("cong", "inc")), key=str):
        R[m]["배치"][str(k)] = round(100 * np.mean(out[m][k]), 1)
    for k in sorted((k for k in out[m] if isinstance(k, tuple) and k[0] == "x"), key=str):
        print("  x,일치", k[1:], f"{100 * np.mean(out[m][k]):.1f}  n={len(out[m][k])}")
    both = [v for v in per[m].values() if True in v and False in v]
    d = [float(v[False]) - float(v[True]) for v in both]; R[m]["문항내_불일치-일치"] = boot(d) + [len(d)]; print("   문항 안 불일치-일치:", R[m]["문항내_불일치-일치"])
json.dump(R, open(f"{RES}/explore_order_v3.json", "w"), ensure_ascii=False, indent=1)

# ---- 총거리 동률 배치: 두 연결이 두 사실의 같은 쪽에 있으면 |a-sa|+|b-so| = |a-so|+|b-sa|.  순서 설명만 차이를 예측한다.
print("\n[총거리] 동률 / 일치 쪽이 짧음 (정렬 매칭은 항상 총거리 최소)")
rows2 = []
for r in Q:
    for c in ("L", "L_sw"):
        ids = r["ctx"][c]; a = next(i for i, d in enumerate(ids) if d in ("LA", "LAs")); b = next(i for i, d in enumerate(ids) if d in ("LB", "LBs"))
        fa, fo = facts(c, r); sa, so = ids.index(fa), ids.index(fo)
        rows2.append(dict(q=r["qid"], t=r["template"], c=c, a=a, b=b, sa=sa, so=so, cong=(a < b) == (sa < so), x=abs(a - sa) - abs(a - so), xB=abs(b - so) - abs(b - sa),
                          tie=abs(a - sa) + abs(b - so) == abs(a - so) + abs(b - sa), y={m: lb(m, r, c) == "공격" for m in P}))
for m in P:
    for t in (True, False):
        s = [z for z in rows2 if z["tie"] == t]
        con = [z["y"][m] for z in s if z["cong"]]; inc = [z["y"][m] for z in s if not z["cong"]]
        print(f" {m} 동률={t}: 일치 {100 * np.mean(con):.1f} (n={len(con)})  불일치 {100 * np.mean(inc):.1f} (n={len(inc)})")
        if t:
            for xv in sorted({z["x"] for z in s}):
                c1 = [z["y"][m] for z in s if z["cong"] and z["x"] == xv]; c2 = [z["y"][m] for z in s if not z["cong"] and z["x"] == xv]
                if c1 and c2: print(f"    x={xv:+d}: 일치 {100 * np.mean(c1):5.1f} (n={len(c1)})  불일치 {100 * np.mean(c2):5.1f} (n={len(c2)})")
            # 문항 쌍 차이: 같은 칸에서 원래/교환
            by = collections.defaultdict(dict)
            for z in s: by[z["q"]][z["cong"]] = z["y"][m]
            d = [float(v[False]) - float(v[True]) for v in by.values() if len(v) == 2]; print("    동률 문항 안 불일치-일치:", boot(d) + [len(d)]); R[m]["동률_불일치-일치"] = boot(d) + [len(d)]
print("\n[회귀] y ~ 불일치 + x + xB (문항 bootstrap).  단위: 불일치 = %p, x, xB = 칸당 %p.  xB = d(B,다른) - d(B,정답)")
qs = sorted({z["q"] for z in rows2}); rng = np.random.default_rng(0); idx = {q: [i for i, z in enumerate(rows2) if z["q"] == q] for q in qs}
X = np.array([[1, float(not z["cong"]), z["x"], z["xB"]] for z in rows2], float)
print(" 상관 (불일치, x, xB):", np.round(np.corrcoef(X[:, 1:].T), 2).tolist())
for m in P:
    y = np.array([z["y"][m] for z in rows2], float); beta = np.linalg.lstsq(X, y, rcond=None)[0]; bs = []
    for _ in range(2000):
        ii = np.concatenate([idx[q] for q in rng.choice(qs, len(qs))]); bs.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0])
    bs = np.array(bs); ci = np.percentile(bs, [2.5, 97.5], axis=0)
    t = {n: [round(100 * beta[j], 1), round(100 * ci[0, j], 1), round(100 * ci[1, j], 1)] for j, n in ((1, "불일치"), (2, "x"), (3, "xB"))}; R[m]["회귀"] = t; print(f" {m}:", t)
json.dump(R, open(f"{RES}/explore_order_v3.json", "w"), ensure_ascii=False, indent=1)
