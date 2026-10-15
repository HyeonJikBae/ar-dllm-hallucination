"""Inside ONE fact: are type-1 (related) wrong rows and type-2 (unrelated) wrong rows in different states?

Same know axis as know_axis_by_k.py (K5 - U direction on train facts, z normalised inside relation with
held-out U / K5 rows; 0 = doesn't know, 1 = knows). Comparisons are within a fact, so fact familiarity,
subject and relation are held fixed; only the paraphrase and the produced answer differ.

  partially known facts (1-4/5) with type-1 AND type-2 wrong rows: z(type1) - z(type2), within-fact AUROC
  same facts that also have a correct row: z(correct) - z(type1), z(correct) - z(type2)
  0/5 facts with type-1 AND type-2 rows: the same contrast when the model does not know the fact (baseline)

Usage:
    PYTHONPATH=sae_analysis python3 sae_analysis/within_fact_types.py --position q_last
"""

import argparse
import collections
import json
import os

import numpy as np

from know_axis_by_k import fact_matrix, load
from twelve_cells import auroc, strat_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--min-freq", type=float, default=0.005)
    ap.add_argument("--min-rel-facts", type=int, default=3)
    ap.add_argument("--out", default="sae_analysis/results_twelve")
    args = ap.parse_args()

    A, corr, reason, sid, rel, k = load(args.position, "final")
    A = A[:, A.mean(0) >= args.min_freq]
    Af = A.astype(np.float32)
    isK5, isU = corr & (k == 5), (k == 0) & ~corr
    facts_all = np.array(sorted(set(sid)))
    res = collections.defaultdict(list)
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        tr = np.isin(sid, rng.permutation(facts_all)[: int(0.6 * len(facts_all))])
        te = ~tr
        fK, XK, rK = fact_matrix(A[tr & isK5], sid[tr & isK5], rel[tr & isK5])
        fU, XU, rU = fact_matrix(A[tr & isU], sid[tr & isU], rel[tr & isU])
        d, _ = strat_diff(XK, rK, XU, rU, args.min_rel_facts)
        s = Af @ d.astype(np.float32)
        z = np.full(len(s), np.nan)
        for r in set(rel):
            mr = te & (rel == r)
            if len(set(sid[mr & isK5])) < args.min_rel_facts or len(set(sid[mr & isU])) < args.min_rel_facts:
                continue
            mu, mk = s[mr & isU].mean(), s[mr & isK5].mean()
            if mk > mu:
                z[mr] = (s[mr] - mu) / (mk - mu)
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in np.where(te & ~np.isnan(z) & (k <= 4))[0]:
            key = "C" if corr[i] else (f"T{reason[i]}" if reason[i] in (1, 2) else None)
            if key:
                by[sid[i]][key].append(z[i])
        for scope, kset in (("부분적으로 아는 fact (1~4/5)", (1, 2, 3, 4)), ("모르는 fact (0/5)", (0,))):
            g12, a12, gc1, gc2, n3 = [], [], [], [], 0
            for f, d_ in by.items():
                kk = k[sid == f][0]
                if kk not in kset or not d_["T1"] or not d_["T2"]:
                    continue
                t1, t2 = np.array(d_["T1"]), np.array(d_["T2"])
                g12.append(t1.mean() - t2.mean())
                a12.append(auroc(t1, t2))
                if d_["C"]:
                    c = np.array(d_["C"])
                    gc1.append(c.mean() - t1.mean())
                    gc2.append(c.mean() - t2.mean())
                    n3 += 1
            res[(scope, "facts (유형1·2 모두)")].append(len(g12))
            res[(scope, "유형1 − 유형2")].append(np.mean(g12))
            res[(scope, "AUROC 유형1 > 유형2")].append(np.mean(a12))
            res[(scope, "유형1 − 유형2 > 0 인 fact 비율")].append(np.mean(np.array(g12) > 0))
            if n3:
                res[(scope, "facts (정답·유형1·유형2 모두)")].append(n3)
                res[(scope, "정답 − 유형1")].append(np.mean(gc1))
                res[(scope, "정답 − 유형2")].append(np.mean(gc2))
    print(f"##### {args.position}: 같은 fact 안 비교 (앎 축 단위: 0 = 모름, 1 = 앎; test fact, {args.splits} splits 평균±sd)")
    out = {}
    for scope in ("부분적으로 아는 fact (1~4/5)", "모르는 fact (0/5)"):
        print(f"\n[{scope}]")
        for (sc, name), v in res.items():
            if sc == scope:
                out[f"{scope} | {name}"] = [float(np.mean(v)), float(np.std(v))]
                fmt = f"{np.mean(v):.0f}" if name.startswith("facts") else f"{np.mean(v):+.3f}±{np.std(v):.3f}"
                print(f"  {name:<28} {fmt}")
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, f"within_fact_types_{args.position}.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
