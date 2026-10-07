"""How well do the knows-vs-doesn't-know latents of PopQA and TriviaQA align?

Score per latent s = f_K - f_U in each dataset (all rows). Reports Pearson/Spearman correlation over
latents active in >=1% of rows in both datasets, top-k overlap, sign agreement and cross-dataset
transfer AUROC of the know axis.

Usage (Dream): FEAT_DIR=sae_analysis/features_dream_sae MODEL_PREFIX=dream_plain PYTHONPATH=sae_analysis \
               python3 sae_analysis/dataset_alignment.py --position prompt_last
"""

import argparse

import numpy as np
from scipy.stats import pearsonr, spearmanr

from three_group import auroc, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last")
    args = ap.parse_args()
    D = {ds: load(ds, args.position) for ds in ("popqa", "triviaqa")}
    S, freq = {}, {}
    for ds, (sid, grp, B) in D.items():
        K, U = grp == "K", np.char.startswith(grp, "U")
        S[ds] = B[K].mean(0) - B[U].mean(0)
        freq[ds] = B.mean(0)
    elig = (freq["popqa"] >= 0.01) & (freq["triviaqa"] >= 0.01)
    a, b = S["popqa"][elig], S["triviaqa"][elig]
    print(f"== {args.position}: eligible latents {elig.sum()}")
    print(f"  corr: pearson {pearsonr(a, b)[0]:.2f}, spearman {spearmanr(a, b)[0]:.2f}")
    for k in (50, 200):
        up = len(set(np.argsort(-S['popqa'])[:k]) & set(np.argsort(-S['triviaqa'])[:k]))
        dn = len(set(np.argsort(S['popqa'])[:k]) & set(np.argsort(S['triviaqa'])[:k]))
        print(f"  top-{k}: K-side overlap {up}/{k}, U-side overlap {dn}/{k} (random ~{k * k / S['popqa'].size:.1f})")
    big = elig & ((np.abs(S["popqa"]) > 0.1) | (np.abs(S["triviaqa"]) > 0.1))
    print(f"  sign agreement (|score|>0.1 in either): {(np.sign(S['popqa'][big]) == np.sign(S['triviaqa'][big])).mean():.2f} (n={big.sum()})")
    for tr, te in (("popqa", "triviaqa"), ("triviaqa", "popqa")):
        sid, grp, B = D[te]
        sc = B @ S[tr]
        K, U = grp == "K", np.char.startswith(grp, "U")
        print(f"  {tr} -> {te}: AUROC K vs U = {auroc(sc[K], sc[U]):.3f}")


if __name__ == "__main__":
    main()
