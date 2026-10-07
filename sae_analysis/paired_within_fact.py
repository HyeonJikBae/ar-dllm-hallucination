"""Within-fact paired comparison: why does the model err on a fact it (sometimes) knows?

For every fact that has both correct rows and rows of a given wrong type, compute per latent
  d_i[j] = mean_{wrong rows}(1[a_j > 0]) - mean_{correct rows}(1[a_j > 0])
and average d_i over facts. Significance: fact-level sign-flip permutation test, BH-FDR.
Because the contrast is inside one fact, fact difficulty and subject identity cancel; only the
paraphrase wording / model state differs.

Usage:
    python3 sae_analysis/paired_within_fact.py --position q_last --types 1 2
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

FEAT = "sae_analysis/features"
DATASETS = ["popqa", "triviaqa"]
NAMES = {1: "related_entity", 2: "unrelated_entity", 3: "subject_copy", 6: "surface_collapse"}


def fact_diffs(position, ds, wrong_type, min_correct):
    meta = [json.loads(l) for l in open(f"{FEAT}/qwen_{ds}_meta.jsonl", encoding="utf-8")]
    acts = np.load(f"{FEAT}/qwen_{ds}_{position}.npy") > 0
    by_fact = {}
    for i, m in enumerate(meta):
        if m["gold_issue"] or m["verdict"] == "uncertain":
            continue
        by_fact.setdefault(m["sample_id"], []).append((i, m))
    diffs = []
    for rows in by_fact.values():
        c = [i for i, m in rows if m["verdict"] == "correct"]
        w = [i for i, m in rows if m["verdict"] == "incorrect" and m["reason"] == wrong_type]
        if len(c) >= min_correct and w:
            diffs.append(acts[w].mean(0) - acts[c].mean(0))
    return np.array(diffs, dtype=np.float32)


def sign_flip_p(d, n_perm, rng):
    """Two-sided p per latent for mean(d) != 0, flipping each fact's sign."""
    obs = np.abs(d.mean(0))
    count = np.zeros(d.shape[1])
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=(d.shape[0], 1)).astype(np.float32)
        count += np.abs((d * s).mean(0)) >= obs
    return (count + 1) / (n_perm + 1)


def bh(p):
    order = np.argsort(p)
    q = np.empty_like(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--types", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--min-correct", type=int, default=1, help="min correct rows per fact")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--min-freq", type=float, default=0.05, help="drop latents active in <5%% of all rows")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="sae_analysis/results_paired")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    for t in args.types:
        per = {ds: fact_diffs(args.position, ds, t, args.min_correct) for ds in DATASETS}
        n_facts = {ds: len(per[ds]) for ds in DATASETS}
        pooled = np.concatenate([per[ds] for ds in DATASETS])
        freq = np.concatenate([np.load(f"{FEAT}/qwen_{ds}_{args.position}.npy") > 0 for ds in DATASETS]).mean(0)
        mean_pooled = pooled.mean(0)
        p = sign_flip_p(pooled, args.n_perm, rng)
        eligible = freq >= args.min_freq
        q = np.ones_like(p)
        q[eligible] = bh(p[eligible])
        print(f"\n[{args.position}] type{t} {NAMES[t]}: facts {n_facts}, "
              f"eligible latents {eligible.sum()}, FDR<0.1: {(q < 0.1).sum()}")
        for direction, sign in (("wrong_up", 1), ("correct_up", -1)):
            score = np.where(eligible, sign * mean_pooled, -np.inf)
            top = np.argsort(-score)[: args.top]
            df = pd.DataFrame({"latent": top, "pooled_diff": sign * mean_pooled[top], "p": p[top], "fdr_q": q[top],
                               "base_freq": freq[top]})
            for ds in DATASETS:
                df[f"diff_{ds}"] = sign * per[ds].mean(0)[top]
            df.to_csv(f"{args.out}/{args.position}_type{t}_{NAMES[t]}_{direction}_min{args.min_correct}.csv", index=False)
            print(f"  {direction}: latent  diff  q  base_freq  popqa  triviaqa")
            for _, r in df.head(6).iterrows():
                print(f"    {int(r.latent):6d} {r.pooled_diff:+.3f} q={r.fdr_q:.3f} f={r.base_freq:.2f} "
                      f"{r.diff_popqa:+.3f} {r.diff_triviaqa:+.3f}")


if __name__ == "__main__":
    main()
