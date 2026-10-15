"""Same error type, different knowledge state: SAE signals of the (state x type) error cells.

States (fact level, k = #correct of the 5 paraphrases, same k as fact_group in 7. final):
  U   0/5 facts   ("doesn't know")
  W   1-4/5 facts ("knows but wrong": the wrong rows of those facts)
  K5  5/5 facts   ("knows": control, correct rows)
Error types 1 (related), 2 (unrelated), 3 (copy), 6 (collapse). 5/5 facts have no wrong rows, so the
3 states x 4 types grid has 8 error cells (U1, U2, U3, U6, W1, W2, W3, W6) plus the K5 control.

Rows: analysis_use rows of 7. final (build_meta_from_final.py: rule_unsure == not analysis_use).
Every comparison is RELATION-STRATIFIED: computed inside each PopQA relation and averaged with weight
min(n_a, n_b) per relation, because the states have very different relation mixes
(U: director/author, W: genre/sport, K5: country/capital). Units of resampling are facts.

  1. cell sizes (rows / facts)
  2. know-axis position of each cell on held-out facts: score = SAE activity . (K5 - U direction learned
     on train facts); 0 = U, 1 = K5
  3. per type t, U_t vs W_t ("same error, different knowledge"):
       - held-out AUROC of a difference-in-means score, plus a label-shuffle null
       - per-latent difference W_t - U_t of fact-mean activity, relation-stratified permutation test, BH-FDR
  4. is the U -> W difference shared across types? correlation of the per-latent differences between types,
     their correlation with the know axis, and latents significant in both types with the same sign
  5. within one state, are the types distinguishable? AUROC type1 vs type2 inside U and inside W

Usage:
    FEAT=sae_analysis/features_refined python3 sae_analysis/twelve_cells.py --position q_last
"""

import argparse
import collections
import json
import os

import numpy as np
from scipy.stats import rankdata

FEAT = os.environ.get("FEAT", "sae_analysis/features_refined")
TYPES = (1, 2, 3, 6)
TYPE_NAMES = {1: "관련", 2: "무관", 3: "복사", 6: "붕괴"}


def load(position):
    meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    ok = np.array([not m["gold_issue"] and m["verdict"] != "uncertain" for m in meta])
    corr = np.array([m["verdict"] == "correct" or m["verdict_suggested"] == "correct" for m in meta])
    use = ok & ~np.array([bool(m.get("rule_unsure")) for m in meta])
    reason = np.array([m["reason"] if m["reason"] is not None else -1 for m in meta])
    sid = np.array([m["sample_id"] for m in meta])
    rel = np.array([m["relation"] for m in meta])
    kf = collections.Counter()
    for s, c, o in zip(sid, corr, ok):
        if o:
            kf[s] += int(c)
    k = np.array([kf.get(s, -1) for s in sid])

    cell = np.full(len(meta), "", dtype=object)
    wrong = use & ~corr & np.isin(reason, TYPES)
    cell[wrong & (k == 0)] = np.char.add("U", reason[wrong & (k == 0)].astype(str))
    w = wrong & (k >= 1) & (k <= 4)
    cell[w] = np.char.add("W", reason[w].astype(str))
    cell[use & corr & (k == 5)] = "K5"
    idx = np.where(cell != "")[0]
    acts = np.load(f"{FEAT}/qwen_popqa_{position}.npy", mmap_mode="r")
    A = np.asarray(acts[idx]) > 0
    return A, cell[idx], sid[idx], rel[idx]


def fact_means(A, sid, rel):
    """Collapse rows to one vector per fact (mean binary activity), so every fact weighs the same."""
    facts, inv = np.unique(sid, return_inverse=True)
    X = np.zeros((len(facts), A.shape[1]), dtype=np.float32)
    np.add.at(X, inv, A.astype(np.float32))
    X /= np.bincount(inv)[:, None]
    frel = np.empty(len(facts), dtype=object)
    frel[inv] = rel
    return facts, X, frel


def strat_weights(rel_a, rel_b, min_n):
    rels = [r for r in sorted(set(rel_a) & set(rel_b)) if (rel_a == r).sum() >= min_n and (rel_b == r).sum() >= min_n]
    w = np.array([min((rel_a == r).sum(), (rel_b == r).sum()) for r in rels], dtype=float)
    return rels, (w / w.sum() if len(w) else w)


def strat_diff(Xa, rel_a, Xb, rel_b, min_n):
    """Relation-stratified difference of means a - b (rows are facts)."""
    rels, w = strat_weights(rel_a, rel_b, min_n)
    d = np.zeros(Xa.shape[1], dtype=np.float64)
    for wi, r in zip(w, rels):
        d += wi * (Xa[rel_a == r].mean(0) - Xb[rel_b == r].mean(0))
    return d, rels


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def strat_auroc(s_a, rel_a, s_b, rel_b, min_n):
    rels, w = strat_weights(rel_a, rel_b, min_n)
    if not rels:
        return np.nan
    return float(sum(wi * auroc(s_a[rel_a == r], s_b[rel_b == r]) for wi, r in zip(w, rels)))


def bh(p):
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1].clip(max=1)
    return q


def perm_test(Xa, rel_a, Xb, rel_b, min_n, n_perm, rng):
    """Two-sided p per latent for the stratified difference, shuffling a/b labels of facts within relation."""
    obs, rels = strat_diff(Xa, rel_a, Xb, rel_b, min_n)
    _, w = strat_weights(rel_a, rel_b, min_n)
    pools = []
    for wi, r in zip(w, rels):
        P = np.concatenate([Xa[rel_a == r], Xb[rel_b == r]])
        pools.append((wi, P, int((rel_a == r).sum()), P.sum(0)))
    cnt = np.zeros(len(obs))
    for _ in range(n_perm):
        d = np.zeros(len(obs))
        for wi, P, na, tot in pools:
            sel = rng.choice(len(P), na, replace=False)
            sa = P[sel].sum(0)
            d += wi * (sa / na - (tot - sa) / (len(P) - na))
        cnt += np.abs(d) >= np.abs(obs) - 1e-12
    return obs, (cnt + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--min-freq", type=float, default=0.01, help="latents active in >= this share of the compared rows")
    ap.add_argument("--min-rel-facts", type=int, default=3, help="facts per side needed for a relation to count")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="sae_analysis/results_twelve")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    A, cell, sid, rel = load(args.position)
    cells = [f"U{t}" for t in TYPES] + [f"W{t}" for t in TYPES] + ["K5"]
    report = {"position": args.position, "sizes": {}, "know_axis": {}, "u_vs_w": {}, "cross_type": {}, "type_within_state": {}}

    # ---------- 1. sizes
    print(f"##### {args.position} | rows {len(cell):,} | {FEAT}")
    print("\n[1] 셀 크기 (행 / fact)")
    print(f"{'유형':<10}{'모두 오답 U':>16}{'아는데 틀림 W':>18}")
    for t in TYPES:
        u, w = cell == f"U{t}", cell == f"W{t}"
        report["sizes"][f"U{t}"] = [int(u.sum()), len(set(sid[u]))]
        report["sizes"][f"W{t}"] = [int(w.sum()), len(set(sid[w]))]
        print(f"{t} {TYPE_NAMES[t]:<8}{u.sum():>9,} / {len(set(sid[u])):>5,}{w.sum():>11,} / {len(set(sid[w])):>5,}")
    k5 = cell == "K5"
    report["sizes"]["K5"] = [int(k5.sum()), len(set(sid[k5]))]
    print(f"모두 정답 K5 (대조): {k5.sum():,} / {len(set(sid[k5])):,}")

    # fact-level matrices per cell (all latents; selection per comparison)
    F = {c: fact_means(A[cell == c], sid[cell == c], rel[cell == c]) for c in cells}
    Uall = cell.astype(str)
    Uall = np.char.startswith(Uall, "U")
    F["U"] = fact_means(A[Uall], sid[Uall], rel[Uall])
    all_facts = np.array(sorted(set(sid)))

    # ---------- 2. know-axis position
    pos = {c: [] for c in cells}
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        tr_f = set(rng.permutation(all_facts)[: int(0.6 * len(all_facts))])
        def split(c, train):
            facts, X, fr = F[c]
            m = np.array([(f in tr_f) == train for f in facts])
            return X[m], fr[m]
        d, _ = strat_diff(*split("K5", True), *split("U", True), args.min_rel_facts)
        s = {c: (split(c, False)[0] @ d, split(c, False)[1]) for c in cells + ["U"]}
        sK, rK = s["K5"]
        sU, rU = s["U"]
        for c in cells:
            sc, rc = s[c]
            num, den = [], []
            for r in set(rc):
                if (rK == r).sum() >= args.min_rel_facts and (rU == r).sum() >= args.min_rel_facts and (rc == r).sum() >= 1:
                    span = sK[rK == r].mean() - sU[rU == r].mean()
                    if span > 0:
                        num.append((rc == r).sum() * (sc[rc == r].mean() - sU[rU == r].mean()) / span)
                        den.append((rc == r).sum())
            pos[c].append(sum(num) / sum(den) if den else np.nan)
    print("\n[2] 앎 축 위치 (0 = 모두 오답 U 평균, 1 = 모두 정답 K5 평균; 관계 보정, test fact, 평균±sd)")
    for t in TYPES:
        u, w = pos[f"U{t}"], pos[f"W{t}"]
        report["know_axis"][f"U{t}"] = [float(np.nanmean(u)), float(np.nanstd(u))]
        report["know_axis"][f"W{t}"] = [float(np.nanmean(w)), float(np.nanstd(w))]
        print(f"  유형 {t} {TYPE_NAMES[t]}:  U {np.nanmean(u):+.2f}±{np.nanstd(u):.2f}   W {np.nanmean(w):+.2f}±{np.nanstd(w):.2f}")
    report["know_axis"]["K5"] = [float(np.nanmean(pos["K5"])), float(np.nanstd(pos["K5"]))]

    # ---------- 3. U_t vs W_t
    print("\n[3] 같은 유형, 다른 앎 상태: U_t vs W_t")
    print("    AUROC = 차이 방향(train fact)으로 test 행을 구분하는 정도, 관계 보정. null = train 라벨을 관계 안에서 섞음")
    diffs, qs, elig_of = {}, {}, {}
    rng = np.random.default_rng(0)
    for t in TYPES:
        cu, cw = f"U{t}", f"W{t}"
        freq = A[(cell == cu) | (cell == cw)].mean(0)
        elig = np.where(freq >= args.min_freq)[0]
        elig_of[t] = elig
        (fu, Xu, ru), (fw, Xw, rw) = F[cu], F[cw]
        Xu_e, Xw_e = Xu[:, elig], Xw[:, elig]
        au, nu = [], []
        for seed in range(args.splits):
            r_ = np.random.default_rng(100 + seed)
            facts = np.concatenate([fu, fw])
            tr_f = set(r_.permutation(facts)[: int(0.6 * len(facts))])
            mu, mw = np.array([f in tr_f for f in fu]), np.array([f in tr_f for f in fw])
            d, _ = strat_diff(Xw_e[mw], rw[mw], Xu_e[mu], ru[mu], args.min_rel_facts)
            # row-level test scores
            te_u = (cell == cu) & ~np.isin(sid, list(tr_f))
            te_w = (cell == cw) & ~np.isin(sid, list(tr_f))
            Su, Sw = A[te_u][:, elig] @ d, A[te_w][:, elig] @ d
            au.append(strat_auroc(Sw, rel[te_w], Su, rel[te_u], args.min_rel_facts))
            # null: shuffle U/W labels among train facts inside each relation
            Xtr = np.concatenate([Xw_e[mw], Xu_e[mu]])
            rtr = np.concatenate([rw[mw], ru[mu]])
            lab = np.r_[np.ones(mw.sum(), bool), np.zeros(mu.sum(), bool)]
            for r in set(rtr):
                m = np.where(rtr == r)[0]
                lab[m] = r_.permutation(lab[m])
            dn, _ = strat_diff(Xtr[lab], rtr[lab], Xtr[~lab], rtr[~lab], args.min_rel_facts)
            nu.append(strat_auroc(A[te_w][:, elig] @ dn, rel[te_w], A[te_u][:, elig] @ dn, rel[te_u], args.min_rel_facts))
        obs, p = perm_test(Xw_e, rw, Xu_e, ru, args.min_rel_facts, args.n_perm, rng)
        q = bh(p)
        diffs[t], qs[t] = obs, q
        fu_, fw_ = A[cell == cu][:, elig].mean(0), A[cell == cw][:, elig].mean(0)
        top = {}
        for name, sg in (("W에서 더 켜짐", 1), ("U에서 더 켜짐", -1)):
            o = [j for j in np.argsort(-sg * obs) if q[j] < 0.05 and sg * obs[j] > 0][: args.top]
            top[name] = [{"latent": int(elig[j]), "diff": float(obs[j]), "q": float(q[j]),
                          "freq_U": float(fu_[j]), "freq_W": float(fw_[j])} for j in o]
        report["u_vs_w"][t] = {
            "facts_U_W": [len(fu), len(fw)], "eligible_latents": int(len(elig)),
            "auroc": [float(np.nanmean(au)), float(np.nanstd(au))], "null_auroc": [float(np.nanmean(nu)), float(np.nanstd(nu))],
            "fdr05": int((q < 0.05).sum()), "fdr05_W_up": int(((q < 0.05) & (obs > 0)).sum()),
            "fdr05_U_up": int(((q < 0.05) & (obs < 0)).sum()), "top": top}
        small = " (W fact 적음: 방향만 참고)" if len(fw) < 150 else ""
        print(f"\n  유형 {t} {TYPE_NAMES[t]}: fact U {len(fu):,} / W {len(fw):,}, 분석 latent {len(elig):,}{small}")
        print(f"    AUROC {np.nanmean(au):.3f}±{np.nanstd(au):.3f}  (null {np.nanmean(nu):.3f}±{np.nanstd(nu):.3f})")
        print(f"    FDR<0.05 latent: {(q < .05).sum()}  (W에서 더 켜짐 {((q < .05) & (obs > 0)).sum()}, U에서 더 켜짐 {((q < .05) & (obs < 0)).sum()})")
        for name, lst in top.items():
            # 괄호: 관계 보정 차이(W-U), 원시 켜짐률 U→W (원시값은 관계 구성 차이로 방향이 다를 수 있음)
            print(f"    {name}: " + "  ".join(f"{x['latent']}({x['diff']:+.2f}; {x['freq_U']:.2f}→{x['freq_W']:.2f})" for x in lst))

    # ---------- 4. shared across types?
    print("\n[4] U→W 차이가 유형 사이에 공통인가 (latent별 차이의 상관, 두 유형 모두 분석된 latent)")
    dk_full, _ = strat_diff(F["K5"][1], F["K5"][2], F["U"][1], F["U"][2], args.min_rel_facts)
    for a, b in [(1, 2), (1, 3), (1, 6), (2, 3), (2, 6), (3, 6)]:
        common = np.intersect1d(elig_of[a], elig_of[b])
        ia, ib = np.searchsorted(elig_of[a], common), np.searchsorted(elig_of[b], common)
        r = float(np.corrcoef(diffs[a][ia], diffs[b][ib])[0, 1])
        both = int(((qs[a][ia] < .05) & (qs[b][ib] < .05) & (np.sign(diffs[a][ia]) == np.sign(diffs[b][ib]))).sum())
        report["cross_type"][f"{a}-{b}"] = {"corr": r, "both_fdr05_same_sign": both, "n_latents": int(len(common))}
        print(f"  유형 {a} vs {b}: 상관 {r:+.2f}, 둘 다 FDR<0.05·같은 방향 latent {both}")
    print("  앎 축(K5-U)과의 상관: " + ", ".join(
        f"유형 {t} {float(np.corrcoef(diffs[t], dk_full[elig_of[t]])[0, 1]):+.2f}" for t in TYPES))
    report["cross_type"]["corr_with_know_axis"] = {t: float(np.corrcoef(diffs[t], dk_full[elig_of[t]])[0, 1]) for t in TYPES}

    # ---------- 5. types within a state
    print("\n[5] 같은 앎 상태 안에서 유형끼리 구분되는가 (AUROC, 관계 보정, test fact)")
    for st in ("U", "W"):
        for a, b in [(1, 2), (1, 3), (1, 6), (2, 3), (2, 6)]:
            ca, cb = f"{st}{a}", f"{st}{b}"
            freq = A[(cell == ca) | (cell == cb)].mean(0)
            elig = np.where(freq >= args.min_freq)[0]
            (fa, Xa, ra), (fb, Xb, rb) = F[ca], F[cb]
            aus = []
            for seed in range(args.splits):
                r_ = np.random.default_rng(200 + seed)
                facts = np.unique(np.concatenate([fa, fb]))
                tr_f = set(r_.permutation(facts)[: int(0.6 * len(facts))])
                ma, mb = np.array([f in tr_f for f in fa]), np.array([f in tr_f for f in fb])
                d, rels = strat_diff(Xa[ma][:, elig], ra[ma], Xb[mb][:, elig], rb[mb], args.min_rel_facts)
                if not rels:
                    continue
                te_a = (cell == ca) & ~np.isin(sid, list(tr_f))
                te_b = (cell == cb) & ~np.isin(sid, list(tr_f))
                aus.append(strat_auroc(A[te_a][:, elig] @ d, rel[te_a], A[te_b][:, elig] @ d, rel[te_b], args.min_rel_facts))
            m = [float(np.nanmean(aus)), float(np.nanstd(aus))] if aus else [np.nan, np.nan]
            report["type_within_state"][f"{ca}_vs_{cb}"] = m
            print(f"  {st}: 유형 {a} vs {b}  AUROC {m[0]:.3f}±{m[1]:.3f}")

    out = os.path.join(args.out, f"twelve_cells_{args.position}.json")
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
