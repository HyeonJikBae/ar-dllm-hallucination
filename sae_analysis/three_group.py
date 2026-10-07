"""Three-state comparison at a pre-answer position: knows / knows-but-wrong / doesn't know.

  K    correct rows                                  (facts with >=1 correct)
  W_t  wrong rows of type t in facts with 1-4 correct ("knows but wrong")
  U    all rows of 0-of-5 facts                      ("doesn't know"); U_t = those of type t

Everything is computed within each dataset (PopQA / TriviaQA separately), with train/test split by
sample_id. A "know axis" d = f_K - f_U (binary activation frequency, train) is built and used to
project test rows; the position of each group is reported on a 0 (=U) .. 1 (=K) scale.

Usage:
    python3 sae_analysis/three_group.py --position q_last
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata

FEAT = os.environ.get("FEAT_DIR", "sae_analysis/features")
PREFIX = os.environ.get("MODEL_PREFIX", "qwen")
TAG = "" if PREFIX == "qwen" else f"_{PREFIX}"
DATASETS = ["popqa", "triviaqa"]


def load(ds, position):
    rows = []
    for tag, suffix in (("kept", ""), ("zero", "_zero")):
        meta = [json.loads(l) for l in open(f"{FEAT}/{PREFIX}_{ds}{suffix}_meta.jsonl", encoding="utf-8")]
        acts = np.load(f"{FEAT}/{PREFIX}_{ds}{suffix}_{position}.npy") > 0
        for m, a in zip(meta, acts):
            if m["gold_issue"]:
                continue
            if tag == "kept":
                g = "K" if m["verdict"] == "correct" else f"W{m['reason']}"
            else:
                g = f"U{m['reason']}"
            rows.append((m["sample_id"], g, a))
    sid = np.array([r[0] for r in rows])
    grp = np.array([r[1] for r in rows])
    B = np.stack([r[2] for r in rows]).astype(np.float32)
    return sid, grp, B


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    for ds in DATASETS:
        sid, grp, B = load(ds, args.position)
        isU = np.char.startswith(grp, "U")
        names = ["K", "W1", "W2", "U1", "U2", "U"]
        print(f"\n===== {ds} / {args.position} =====")
        print("rows per group:", {n: int((grp == n).sum()) if n != "U" else int(isU.sum()) for n in names},
              "| facts:", len(set(sid)))

        facts = np.array(sorted(set(sid)))
        res = {n: [] for n in names}
        au = {"K_vs_U": [], "K_vs_W1": [], "K_vs_W2": [], "W1_vs_U": [], "W2_vs_U": []}
        for seed in range(args.splits):
            rng = np.random.default_rng(seed)
            tr_f = set(rng.permutation(facts)[: int(0.6 * len(facts))])
            tr = np.isin(sid, list(tr_f))
            te = ~tr
            fK = B[tr & (grp == "K")].mean(0)
            fU = B[tr & isU].mean(0)
            d = fK - fU
            score = B @ d
            mem = {"K": grp == "K", "W1": grp == "W1", "W2": grp == "W2",
                   "U1": grp == "U1", "U2": grp == "U2", "U": isU}
            m = {n: score[te & mem[n]].mean() if (te & mem[n]).any() else np.nan for n in names}
            den = m["K"] - m["U"]
            for n in names:
                res[n].append((m[n] - m["U"]) / den)
            s = lambda n: score[te & mem[n]]
            au["K_vs_U"].append(auroc(s("K"), s("U")))
            au["K_vs_W1"].append(auroc(s("K"), s("W1")))
            au["K_vs_W2"].append(auroc(s("K"), s("W2")))
            au["W1_vs_U"].append(auroc(s("W1"), s("U")))
            au["W2_vs_U"].append(auroc(s("W2"), s("U")))
        print("position on know-axis (0 = doesn't know, 1 = knows), mean ± sd over splits:")
        for n in names:
            print(f"  {n:3s} {np.nanmean(res[n]):+.2f} ± {np.nanstd(res[n]):.2f}")
        print("AUROC (test, row-level):", {k: f"{np.nanmean(v):.3f}" for k, v in au.items()})

        # features: W differs from K AND from U (neither state), selected on one train split
        rng = np.random.default_rng(0)
        tr = np.isin(sid, list(set(rng.permutation(facts)[: int(0.6 * len(facts))])))
        te = ~tr
        for w in ("W1", "W2"):
            fK, fW, fU = B[tr & (grp == "K")].mean(0), B[tr & (grp == w)].mean(0), B[tr & isU].mean(0)
            tK, tW, tU = (B[te & (grp == "K")].mean(0), B[te & (grp == w)].mean(0), B[te & isU].mean(0))
            for label, score_tr, score_te in (
                ("W-specific up (W > K and W > U)", np.minimum(fW - fK, fW - fU), np.minimum(tW - tK, tW - tU)),
                ("W-specific down (W < K and W < U)", np.minimum(fK - fW, fU - fW), np.minimum(tK - tW, tU - tW)),
            ):
                top = np.argsort(-score_tr)[: args.top]
                print(f"\n  [{w}] {label}: latent  train_score  test_score  fK/fW/fU(test)")
                for j in top[:6]:
                    print(f"    {j:6d}  {score_tr[j]:+.3f}  {score_te[j]:+.3f}  {tK[j]:.2f}/{tW[j]:.2f}/{tU[j]:.2f}")


if __name__ == "__main__":
    main()
