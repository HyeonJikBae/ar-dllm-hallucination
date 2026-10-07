"""Feature tables for the three states: knows (K) / knows-but-wrong (W_t) / doesn't know (U).

For each wrong type t in {1, 2}, latents are ranked (on train facts, min over datasets) by how
specific they are to one state, i.e. higher than BOTH other states:
  K-specific       min(fK - fW, fK - fU)
  U-specific       min(fU - fK, fU - fW)
  W-specific up    min(fW - fK, fW - fU)
  W-specific down  min(fK - fW, fU - fW)     (lower on W than on both K and U)
f = fraction of rows where the latent is active (>0). Split by sample_id: 50/10/40 train/val/test.

Usage:
    python3 sae_analysis/three_group_tables.py --position q_last
"""

import argparse
import os

import numpy as np

from three_group import DATASETS, TAG, load

TYPES = {1: "related_entity", 2: "unrelated_entity"}
KINDS = {
    "K_specific": lambda k, w, u: np.minimum(k - w, k - u),
    "U_specific": lambda k, w, u: np.minimum(u - k, u - w),
    "W_specific_up": lambda k, w, u: np.minimum(w - k, w - u),
    "W_specific_down": lambda k, w, u: np.minimum(k - w, u - w),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sae_analysis")
    args = ap.parse_args()

    data = {}
    for ds in DATASETS:
        sid, grp, B = load(ds, args.position)
        rng = np.random.default_rng(args.seed)
        facts = rng.permutation(sorted(set(sid)))
        a, b = int(0.5 * len(facts)), int(0.6 * len(facts))
        split = {"train": set(facts[:a]), "val": set(facts[a:b]), "test": set(facts[b:])}
        data[ds] = (sid, grp, B, {k: np.isin(sid, list(v)) for k, v in split.items()})

    def freqs(ds, split, group_mask_fn):
        sid, grp, B, masks = data[ds]
        m = masks[split] & group_mask_fn(grp)
        return B[m].mean(0), int(m.sum())

    lines = [f"# Three-state tables ({args.position}): K=knows, W=knows-but-wrong, U=doesn't know\n",
             "Cells are 'f_K / f_W / f_U' (activation frequency) per split; score = specificity (see script docstring).\n"]
    for t, name in TYPES.items():
        isK = lambda g: g == "K"
        isW = lambda g, t=t: g == f"W{t}"
        isU = lambda g: np.char.startswith(g, "U")
        sizes = {ds: {s: (freqs(ds, s, isK)[1], freqs(ds, s, isW)[1], freqs(ds, s, isU)[1]) for s in ("train", "val", "test")}
                 for ds in DATASETS}
        lines.append(f"\n## type{t} {name}\n")
        lines.append("rows (K/W/U) per split: " + "; ".join(f"{ds} " + ", ".join(f"{s} {v}" for s, v in sizes[ds].items()) for ds in DATASETS) + "\n")
        for kind, fn in KINDS.items():
            scores = {}
            for ds in DATASETS:
                for s in ("train", "val", "test"):
                    fk, fw, fu = freqs(ds, s, isK)[0], freqs(ds, s, isW)[0], freqs(ds, s, isU)[0]
                    scores[(ds, s)] = (fn(fk, fw, fu), fk, fw, fu)
            sel = np.minimum(*[scores[(ds, "train")][0] for ds in DATASETS])
            top = np.argsort(-sel)[: args.top]
            lines.append(f"\n### {kind}\n")
            lines.append("| latent | train(min) | val popqa | val trivia | test popqa | test trivia | test K/W/U popqa | test K/W/U trivia |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for j in top:
                cell = lambda ds, s: scores[(ds, s)][0][j]
                kwu = lambda ds: "/".join(f"{scores[(ds, 'test')][i][j]:.2f}" for i in (1, 2, 3))
                lines.append(f"| {j} | {sel[j]:+.3f} | {cell('popqa', 'val'):+.3f} | {cell('triviaqa', 'val'):+.3f} | "
                             f"{cell('popqa', 'test'):+.3f} | {cell('triviaqa', 'test'):+.3f} | {kwu('popqa')} | {kwu('triviaqa')} |")
    path = os.path.join(args.out, f"three_state_tables{TAG}_{args.position}.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
