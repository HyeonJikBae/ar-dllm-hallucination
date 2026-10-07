"""Is the within-fact "wrong" signal just a loss of the know-state, or something else?

Per model / dataset / position / wrong type t (hidden states, standardized on train rows):
  u        know axis = mean(K rows) - mean(U rows)                       (K correct rows, U 0-of-5 rows)
  w        within-fact probe (logistic regression, fact-centered) for W-vs-K
Reported (20 fact-level 60/40 splits, test facts, within-fact AUROC = P(score_W > score_K) per fact):
  full     probe w
  axis     score = -x.u  (wrongness predicted only by loss of the know axis)
  resid    probe retrained after projecting u out of every row
  cos(w,u) cosine between the probe direction and the know axis
  by k     full-probe AUROC split by k = #correct paraphrases of the fact
  consist  mean leave-one-out cosine between a fact's (mean W - mean K) vector and the mean of the
           others' (shared "wrong direction" across facts); null = labels shuffled inside facts

Usage:
    python3 sae_analysis/know_axis_removal.py
"""

import argparse
import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probe_within_fact import MODELS, center, load_meta

warnings.filterwarnings("ignore")


def fact_aucs(scores, y, sid):
    out = {}
    for f in np.unique(sid):
        m = sid == f
        w, k = scores[m & (y == 1)], scores[m & (y == 0)]
        if len(w) and len(k):
            d = w[:, None] - k[None, :]
            out[f] = ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size
    return out


def fit_lr(X, y, sid, C):
    Xc = center(X, sid)
    sc = StandardScaler().fit(Xc)
    clf = LogisticRegression(C=C, max_iter=500).fit(sc.transform(Xc), y)
    return clf, sc


def consistency(Xs, y, sid, rng, null=False):
    facts = np.unique(sid)
    yy = y.copy()
    if null:
        for f in facts:
            m = np.where(sid == f)[0]
            yy[m] = rng.permutation(y[m])
    deltas = []
    for f in facts:
        m = sid == f
        if (yy[m] == 1).any() and (yy[m] == 0).any():
            deltas.append(Xs[m & (yy == 1)].mean(0) - Xs[m & (yy == 0)].mean(0))
    D = np.array(deltas)
    tot = D.sum(0)
    cos = []
    for d in D:
        o = tot - d
        cos.append(d @ o / (np.linalg.norm(d) * np.linalg.norm(o) + 1e-9))
    return float(np.mean(cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--datasets", nargs="+", default=["popqa", "triviaqa"])
    ap.add_argument("--out", default="sae_analysis/know_axis_removal.json")
    args = ap.parse_args()

    results = []
    print(f"{'model':5s} {'data':8s} {'pos':11s} {'t':1s} {'facts':>5s} | {'full':>5s} {'axis':>5s} {'resid':>5s} {'cos(w,u)':>8s} | "
          f"{'k=1':>9s} {'k=2':>9s} {'k=3':>9s} {'k=4':>9s} | {'consist':>7s} {'null':>6s}")
    for model, cfg in MODELS.items():
        for ds in args.datasets:
            meta = load_meta(cfg["meta"].format(ds=ds))
            kk = {}
            for m in meta:
                if m["group"] == "kept" and not m["gold_issue"] and m["verdict"] == "correct":
                    kk[m["sample_id"]] = kk.get(m["sample_id"], 0) + 1
            for pos in ("q_last", "prompt_last"):
                H = np.load(cfg["hid"].format(ds=ds, pos=pos)).astype(np.float32)
                zero_idx = np.array([i for i, m in enumerate(meta) if m["group"] == "zero" and not m["gold_issue"]])
                for t in (1, 2):
                    idx, y, sid = [], [], []
                    for i, m in enumerate(meta):
                        if m["group"] != "kept" or m["gold_issue"]:
                            continue
                        if m["verdict"] == "correct":
                            idx.append(i); y.append(0); sid.append(m["sample_id"])
                        elif m["verdict"] == "incorrect" and m["reason"] == t:
                            idx.append(i); y.append(1); sid.append(m["sample_id"])
                    idx, y, sid = np.array(idx), np.array(y), np.array(sid)
                    paired = [f for f in np.unique(sid) if (y[sid == f] == 0).any() and (y[sid == f] == 1).any()]
                    mk = np.isin(sid, paired)
                    idx, y, sid = idx[mk], y[mk], sid[mk]
                    X, XU = H[idx], H[zero_idx]
                    facts = np.array(paired)
                    agg = {k: [] for k in ("full", "axis", "resid", "cos")}
                    byk = {k: [] for k in (1, 2, 3, 4)}
                    for seed in range(args.splits):
                        rng = np.random.default_rng(seed)
                        tr_f = set(rng.permutation(facts)[: int(0.6 * len(facts))])
                        tr = np.array([s in tr_f for s in sid]); te = ~tr
                        if len(np.unique(y[tr])) < 2:
                            continue
                        sc = StandardScaler().fit(np.vstack([X[tr], XU]))
                        Xs, XUs = sc.transform(X), sc.transform(XU)
                        u = Xs[tr & (y == 0)].mean(0) - XUs.mean(0)
                        u /= np.linalg.norm(u)
                        clf, scl = fit_lr(Xs[tr], y[tr], sid[tr], args.C)
                        w = clf.coef_[0] / scl.scale_
                        a = fact_aucs(Xs[te] @ w, y[te], sid[te])
                        agg["full"].append(np.mean(list(a.values())))
                        agg["cos"].append(float(w @ u / np.linalg.norm(w)))
                        agg["axis"].append(np.mean(list(fact_aucs(-(Xs[te] @ u), y[te], sid[te]).values())))
                        Xr = Xs - np.outer(Xs @ u, u)
                        clf2, scl2 = fit_lr(Xr[tr], y[tr], sid[tr], args.C)
                        w2 = clf2.coef_[0] / scl2.scale_
                        agg["resid"].append(np.mean(list(fact_aucs(Xr[te] @ w2, y[te], sid[te]).values())))
                        for f, v in a.items():
                            if kk.get(f, 0) in byk:
                                byk[kk[f]].append(v)
                    Xs_all = StandardScaler().fit_transform(X)
                    rng = np.random.default_rng(0)
                    cons = consistency(Xs_all, y, sid, rng)
                    null = float(np.mean([consistency(Xs_all, y, sid, np.random.default_rng(s), null=True) for s in range(10)]))
                    row = {"model": model, "dataset": ds, "pos": pos, "type": t, "facts": len(paired),
                           **{k: float(np.mean(v)) for k, v in agg.items()},
                           "by_k": {k: [float(np.mean(v)) if v else None, len(v)] for k, v in byk.items()},
                           "consistency": cons, "consistency_null": null}
                    results.append(row)
                    nk = {k: sum(kk.get(f, 0) == k for f in paired) for k in (1, 2, 3, 4)}
                    row["facts_by_k"] = nk
                    kf = lambda k: (f"{row['by_k'][k][0]:.2f}[{nk[k]:>3d}]" if row['by_k'][k][0] is not None else "   -     ")
                    print(f"{model:5s} {ds:8s} {pos:11s} {t:<1d} {len(paired):5d} | {row['full']:.3f} {row['axis']:.3f} {row['resid']:.3f} {row['cos']:+8.3f} | "
                          f"{kf(1)} {kf(2)} {kf(3)} {kf(4)} | {cons:7.3f} {null:6.3f}", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
