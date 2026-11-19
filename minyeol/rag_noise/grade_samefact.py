"""patch_samefact.py 채점 (PREREG_poison.md v6).  usage: python grade_samefact.py     출력: results/summary_samefact.json
반영 = 주는 입력의 정답 (I1>I2: aX, I2>I1: aY), 두 방향 평균.  층 구간: 초반 decoder 0-8, 중반 10-18, 후반 20-27.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"; B = 10000
LAYERS = ["emb"] + [str(l) for l in range(0, 28, 2)] + ["27"]
BANDS = {"초반": range(1, 6), "중반": range(6, 11), "후반": range(11, 16)}
SITES = ["fact", "qall", "last", "fq", "ctrl_f", "ctrl_fq"]


def rates(rec, tm, site, k):
    """[16] 층별 비율 (두 방향 평균).  k = 0 반영, 1 유지, 2 Z값."""
    out = np.zeros(len(LAYERS))
    for d, (donor, recv) in (("I1>I2", (0, 1)), ("I2>I1", (1, 0))):
        tgt = rec["cand"][[donor, recv, 2][k]]
        out += 0.5 * (np.array(rec["patch"][tm][d][site]) == tgt)
    return out


def boot(x, rng):
    """x [n]: 평균, 95% CI, 양측 p."""
    W = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=B) / len(x); bs = W @ x
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))


def holm(ps):
    o = np.argsort(ps); adj = np.zeros(len(ps)); run = 0
    for i, j in enumerate(o): run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1, run)
    return adj


def fmt(m, lo, hi): return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    rng = np.random.default_rng(0)
    Q = {r["qid"]: r for r in map(json.loads, open(f"{RES}/samefact_qwen.jsonl"))}
    D = {r["qid"]: r for r in map(json.loads, open(f"{RES}/samefact_dream.jsonl"))}
    qs = sorted(q for q in D if Q[q]["ok"]["ar"] and all(D[q]["ok"].values()))
    S = dict(n=len(Q), qwen_ok=sum(r["ok"]["ar"] for r in Q.values()), dream_ok=sum(all(r["ok"].values()) for r in D.values()), n_main=len(qs))
    print(f"쌍 {S['n']}  Qwen I1·I2 정답 {S['qwen_ok']}  그중 Dream (t0, tc) 정답 {S['dream_ok']}  주 집합 {len(qs)}")
    MT = [("qwen", "ar", Q), ("dream", "t0", D), ("dream", "tc", D)]
    A = {(m, tm, s): np.stack([rates(R[q], tm, s, 0) for q in qs]) for m, tm, R in MT for s in SITES}
    S["curves"] = {f"{m}/{tm}/{s}": (A[m, tm, s].mean(0) * 100).round(1).tolist() for m, tm, _ in MT for s in SITES}
    print("\n반영 % (층: " + " ".join(LAYERS) + ")")
    for m, tm, _ in MT:
        for s in SITES: print(f"  {m:5s} {tm} {s:7s} " + " ".join(f"{v:5.1f}" for v in A[m, tm, s].mean(0) * 100))
    band = lambda a, b: a[:, list(BANDS[b])].mean(1)
    S["H6.1"], S["H6.2"] = {}, {}
    for m, tm, _ in MT:
        res = {b: boot(band(A[m, tm, "fact"], b) - band(A[m, tm, "ctrl_f"], b), rng) for b in BANDS}
        adj = holm([res[b][3] for b in BANDS])
        S["H6.1"][f"{m}/{tm}"] = {b: dict(diff=fmt(*res[b][:3]), p_holm=round(float(p), 4)) for b, p in zip(BANDS, adj)}
        joint = {}
        for b in BANDS:
            mx = np.maximum(A[m, tm, "fact"], A[m, tm, "qall"])
            joint[b] = dict(fq_minus_max=fmt(*boot(band(A[m, tm, "fq"], b) - band(mx, b), rng)[:3]),
                            fq_minus_ctrl=fmt(*boot(band(A[m, tm, "fq"], b) - band(A[m, tm, "ctrl_fq"], b), rng)[:3]))
        S["H6.2"][f"{m}/{tm}"] = joint
    S["H6.3"] = {b: dict(fact_ctrl_tc_minus_t0=fmt(*boot((band(A["dream", "tc", "fact"], b) - band(A["dream", "tc", "ctrl_f"], b))
                                                          - (band(A["dream", "t0", "fact"], b) - band(A["dream", "t0", "ctrl_f"], b)), rng)[:3]),
                         last_tc_minus_t0=fmt(*boot(band(A["dream", "tc", "last"], b) - band(A["dream", "t0", "last"], b), rng)[:3])) for b in BANDS}
    # 기술: 주는 입력의 정답 사실이 대상 연결 문장 (A) 뒤에 있는가 (I1>I2: X 사실 칸 p, I2>I1: Y 사실 칸 q)
    S["order_split"] = {}
    for m, tm, R in MT:
        o = {True: [], False: []}
        for q in qs:
            r = R[q]
            for d, slot, tgt in (("I1>I2", r["p_slot"], r["cand"][0]), ("I2>I1", r["q_slot"], r["cand"][1])):
                o[slot > r["la_slot"]].append([np.mean([t == tgt for t in np.array(r["patch"][tm][d]["fact"])[list(BANDS[b])]]) for b in BANDS])
        S["order_split"][f"{m}/{tm}"] = {("연결 뒤" if k else "연결 앞"): dict(n=len(v), **{b: round(float(np.mean([x[i] for x in v])) * 100, 1)
                                                                                for i, b in enumerate(BANDS)}) for k, v in o.items()}
    for k in ("H6.1", "H6.2", "H6.3", "order_split"):
        print(f"\n{k}"); [print(" ", kk, vv) for kk, vv in S[k].items()]
    json.dump(S, open(f"{RES}/summary_samefact.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
