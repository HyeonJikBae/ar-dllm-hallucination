"""patch_bind.py 결과 채점 (PREREG_poison.md v4, v4.1).
usage: python grade_bindpatch.py [bindpatch_pilot|bindpatch]     출력: results/summary_<data>.json
층별 범주 비율 (두 방향 평균) 을 모델 x 자리마다 출력.  파일럿은 link 관문과 L* 를 정하고, 본 실행은 L* 에서 H1-H3 를 계산.
분석 집합: common = 두 모델 모두 src·dst 정답.  strict = common 중 link 임베딩 층 개입이 두 모델·두 방향 모두 개체.
  파일럿 link 관문 (개체 >= 80%) 을 두 모델 모두 넘으면 주 = common, 아니면 주 = strict (v4.1).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"
CATS = ["유지", "개체", "위치", "복사", "주는쪽", "기타"]
ROLE = {"s2d": dict(bY=0, bX=1, bZ=2, aX=3, aY=4, aZ=4), "d2s": dict(aX=0, aY=1, aZ=2, bY=3, bX=4, bZ=4)}
VALS = ["aX", "aY", "aZ", "bX", "bY", "bZ"]; B = 10000; SITES = ["link", "qhead", "qall", "last", "masks"]


def load(model, data):
    return {r["qid"]: r for r in map(json.loads, open(f"{RES}/patch_{model}_{data}.jsonl"))}


def cats(rec, site):
    """[29, 6] 범주 비율 (두 방향 평균)."""
    out = np.zeros((len(rec["s2d"][site]["argmax"]), len(CATS)))
    for d in ("s2d", "d2s"):
        m = {rec["cand"][VALS.index(k)]: c for k, c in ROLE[d].items()}
        for l, t in enumerate(rec[d][site]["argmax"]): out[l, m.get(t, 5)] += 0.5
    return out


def boot(a, rng):
    """a [n, ...] 의 평균과 쌍 단위 bootstrap 95% CI."""
    W = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=B) / len(a)
    bs = np.tensordot(W, a, axes=1); return a.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def ci(x, lo, hi): return f"{100 * x:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def curves(R, qs, verbose):
    A, out = {}, {}
    for m in R:
        for site in [s for s in SITES if s in R[m][qs[0]]["s2d"]]:
            A[m, site] = np.stack([cats(R[m][q], site) for q in qs]); out[f"{m}/{site}"] = (A[m, site].mean(0) * 100).round(1).tolist()
            if verbose:
                print(f"\n{m} / {site}  (층: " + "  ".join(CATS) + ")")
                for l, row in enumerate(A[m, site].mean(0)): print(f"  {'emb' if l == 0 else l - 1:>3} " + " ".join(f"{100 * v:5.1f}" for v in row))
    return A, out


def tests(R, qs, A, Ls, rng):
    T = {}
    for m in R:
        for s in ("last", "qall"):
            L = Ls[f"{m}/{s}"]; a = A[m, s][:, L]
            d = np.stack([a[:, 1] - a[:, 3], a[:, 1] - a[:, 2], a[:, 1], a[:, 2], a[:, 3], a[:, 1] + a[:, 2]], 1)
            mu, lo, hi = boot(d, rng)
            t = dict(layer=L - 1, **{k: ci(mu[i], lo[i], hi[i]) for i, k in enumerate(["H1_개체-복사", "H2_개체-위치", "개체", "위치", "복사", "개체+위치"])})
            ep = A[m, s][:, :, 1:3].sum(-1).mean(0); cp = A[m, s][:, :, 3].mean(0)
            t["처음_개체+위치>=50_층"] = next((l - 1 for l in range(len(ep)) if ep[l] >= .5), None)
            t["복사>개체+위치_첫층"] = next((l - 1 for l in range(len(ep)) if cp[l] > ep[l]), None)
            T[f"{m}/{s}"] = t
    for s in ("last", "qall"):
        Lq, Ld = Ls[f"qwen/{s}"], Ls[f"dream/{s}"]
        d = np.stack([A["dream", s][:, Ld, k] - A["qwen", s][:, Lq, k] for k in (1, 2)] +
                     [A["dream", s][:, Ld, 1:3].sum(-1) - A["qwen", s][:, Lq, 1:3].sum(-1)], 1)
        mu, lo, hi = boot(d, rng); T[f"dream-qwen/{s}"] = {k: ci(mu[i], lo[i], hi[i]) for i, k in enumerate(["개체", "위치", "개체+위치"])}
    T["관계틀별_last_L*_개체"] = {m: [round(float(A[m, "last"][[i for i, q in enumerate(qs) if R[m][q]["template"] == t], Ls[f"{m}/last"], 1].mean() * 100), 1)
                                 for t in range(9) if any(R[m][q]["template"] == t for q in qs)] for m in R}
    return T


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else "bindpatch"; rng = np.random.default_rng(0)
    R = {m: load(m, data) for m in ("qwen", "dream")}
    common = sorted(q for q in R["qwen"] if R["qwen"][q]["ok"] and q in R["dream"] and R["dream"][q]["ok"])
    strict = [q for q in common if all(cats(R[m][q], "link")[0, 1] == 1 for m in R)]
    S = dict(n=len(R["qwen"]), n_ok={m: sum(r["ok"] for r in R[m].values()) for m in R}, n_common=len(common), n_strict=len(strict))
    S["link_emb_개체"] = {m: round(float(np.mean([cats(R[m][q], "link")[0, 1] for q in common])) * 100, 1) for m in R}
    print(f"[{data}] 쌍 {S['n']}  src·dst 모두 정답 {S['n_ok']}  공통 {len(common)}  strict {len(strict)}  link 임베딩 개체 {S['link_emb_개체']}")
    if data.endswith("pilot"):
        A, S["curves"] = curves(R, common, True)
        S["Lstar"] = {f"{m}/{s}": int(A[m, s][:, :, 1:3].sum(-1).mean(0).argmax()) for m in R for s in ("last", "qall")}
        S["use_strict"] = any(v < 80 for v in S["link_emb_개체"].values())
        print("\nL* (배열 번호, 0 = 임베딩):", S["Lstar"], " 주 분석 집합:", "strict" if S["use_strict"] else "common")
    else:
        P = json.load(open(f"{RES}/summary_{data}_pilot.json")); Ls = P["Lstar"]
        sets = [("strict", strict), ("common", common)] if P["use_strict"] else [("common", common), ("strict", strict)]
        S["primary"] = sets[0][0]
        for k, (name, qs) in enumerate(sets):
            A, S[f"curves_{name}"] = curves(R, qs, k == 0)
            S[f"tests_{name}"] = tests(R, qs, A, Ls, rng)
            print(f"\n== {name} ({len(qs)} 쌍{', 주' if k == 0 else ', 보조'})")
            for key, v in S[f"tests_{name}"].items(): print(" ", key, v)
        for m in R:                                                          # 모델별 집합 (각자 맞힌 쌍)
            own = [q for q, r in R[m].items() if r["ok"]]
            S[f"curves_own_{m}"] = {s: (np.stack([cats(R[m][q], s) for q in own]).mean(0) * 100).round(1).tolist() for s in SITES if s in R[m][own[0]]["s2d"]}
    json.dump(S, open(f"{RES}/summary_{data}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
