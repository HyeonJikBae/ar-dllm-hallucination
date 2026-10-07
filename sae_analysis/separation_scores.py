"""Latent separation scores between correct answers and each wrong-answer type.

Follows Ferrando et al. (ICLR 2025, "Do I Know This Entity?"):
  f_g[j]   = fraction of rows in group g where SAE latent j is active (> 0)
  s_type   = f_type    - f_correct   (latents more active on the wrong type)
  s_correct= f_correct - f_type      (latents more active on correct answers)
Scores are computed per dataset on a train split (split by sample_id so the five
paraphrases of one fact never straddle splits); generality = min over datasets;
val/test scores are reported for the selected latents.

Usage:
    python3 sae_analysis/separation_scores.py --position q_last
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

FEAT = "sae_analysis/features"
DATASETS = ["popqa", "triviaqa"]
TYPES = {1: "related_entity", 2: "unrelated_entity", 3: "subject_copy", 6: "surface_collapse"}
MIN_TRAIN_ROWS = 20


def load(position):
    data = {}
    for ds in DATASETS:
        meta = [json.loads(l) for l in open(f"{FEAT}/qwen_{ds}_meta.jsonl", encoding="utf-8")]
        acts = np.load(f"{FEAT}/qwen_{ds}_{position}.npy", mmap_mode="r")
        keep = np.array([not m["gold_issue"] for m in meta])
        label = np.array(
            ["correct" if m["verdict"] == "correct" else f"type{m['reason']}" for m in meta]
        )
        sid = np.array([m["sample_id"] for m in meta])
        data[ds] = ((np.asarray(acts) > 0)[keep], label[keep], sid[keep])
    return data


def split_ids(sids, rng, frac=(0.5, 0.1, 0.4)):
    ids = np.array(sorted(set(sids)))
    rng.shuffle(ids)
    a, b = int(frac[0] * len(ids)), int((frac[0] + frac[1]) * len(ids))
    return {"train": set(ids[:a]), "val": set(ids[a:b]), "test": set(ids[b:])}


def freq(binary, mask):
    return binary[mask].mean(0) if mask.sum() else np.full(binary.shape[1], np.nan), int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="q_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="sae_analysis/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = load(args.position)
    rng = np.random.default_rng(args.seed)
    splits = {ds: split_ids(data[ds][2], rng) for ds in DATASETS}

    def f(ds, split, lab):
        b, label, sid = data[ds]
        mask = (label == lab) & np.isin(sid, list(splits[ds][split]))
        return freq(b, mask)

    for t, name in TYPES.items():
        lab = f"type{t}"
        rows = {}
        n = {}
        for split in ("train", "val", "test"):
            for ds in DATASETS:
                fc, nc = f(ds, split, "correct")
                ft, nt = f(ds, split, lab)
                rows[(split, ds)] = ft - fc  # + : more active on wrong type
                n[(split, ds)] = (nt, nc)
        ok = [ds for ds in DATASETS if n[("train", ds)][0] >= MIN_TRAIN_ROWS]
        print(f"[{args.position}] {lab} {name}: train n_type/n_correct "
              + ", ".join(f"{ds}={n[('train', ds)]}" for ds in DATASETS) + f" -> min over {ok}")
        if not ok:
            continue
        for direction, sign in (("wrong_up", 1), ("correct_up", -1)):
            s_train = np.stack([sign * rows[("train", ds)] for ds in ok])
            score = s_train.min(0)
            top = np.argsort(-score)[: args.top]
            df = pd.DataFrame({"latent": top, "min_train_score": score[top]})
            for split in ("train", "val", "test"):
                for ds in DATASETS:
                    df[f"{split}_{ds}"] = sign * rows[(split, ds)][top]
            df.to_csv(f"{args.out}/{args.position}_{lab}_{name}_{direction}.csv", index=False)
            print(f"  {direction}: top5 latent/min_train/test_popqa/test_triviaqa")
            for _, r in df.head(5).iterrows():
                print(f"    {int(r.latent):6d}  {r.min_train_score:+.3f}  {r.test_popqa:+.3f}  {r.test_triviaqa:+.3f}")


if __name__ == "__main__":
    main()
