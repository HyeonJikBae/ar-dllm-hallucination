"""Common wrong-answer SAE latents on the refined PopQA labels (7. final).

Groups: correct (verdict correct or suggested correct) vs wrong types 1/2/3/6.
f_g[j]  = fraction of rows in group g where latent j > 0
s_type  = f_type - f_correct (Ferrando et al.); common score = min over the wrong types.

Two controls:
  raw     all facts (confounded by relation / entity popularity)
  paired  facts that contain BOTH correct and wrong rows among their 5 paraphrases: the per-fact
          difference of activation rates is averaged over facts, so the entity is held fixed.

Usage:
    python3 sae_analysis/separation_refined.py --position prompt_last
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

FEAT = "sae_analysis/features_refined"
TYPES = {1: "related", 2: "unrelated", 3: "copy", 6: "collapse"}
MIN_ROWS = 20


def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default="prompt_last", choices=["q_last", "prompt_last"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="sae_analysis/results_refined")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = [json.loads(l) for l in open(f"{FEAT}/qwen_popqa_meta.jsonl", encoding="utf-8")]
    acts = np.load(f"{FEAT}/qwen_popqa_{args.position}.npy", mmap_mode="r")
    keep = np.array([not m["gold_issue"] for m in meta])
    lab = []
    for m in meta:
        if m["verdict"] == "correct" or m["verdict_suggested"] == "correct":
            lab.append("correct")
        elif m["reason"] in TYPES:
            lab.append(f"t{m['reason']}")
        else:
            lab.append("other")
    lab = np.array(lab)
    sid = np.array([m["sample_id"] for m in meta])
    rel = np.array([m["relation"] for m in meta])
    use = keep & (lab != "other")
    A = np.asarray(acts)[use]
    lab, sid, rel = lab[use], sid[use], rel[use]
    meta = [m for m, u in zip(meta, use) if u]
    B = A > 0
    print(f"[{args.position}] rows {len(lab)}  groups {dict(zip(*np.unique(lab, return_counts=True)))}")

    rng = np.random.default_rng(args.seed)
    ids = np.array(sorted(set(sid)))
    rng.shuffle(ids)
    a, b = int(0.5 * len(ids)), int(0.6 * len(ids))
    split = {"train": set(ids[:a]), "val": set(ids[a:b]), "test": set(ids[b:])}
    sm = {k: np.isin(sid, list(v)) for k, v in split.items()}

    def freq(mask):
        return B[mask].mean(0) if mask.sum() else np.full(B.shape[1], np.nan)

    def paired(mask, t):
        """mean over facts of (rate among type-t rows - rate among correct rows)."""
        diffs, cnt = [], 0
        for s in np.unique(sid[mask]):
            r = np.where((sid == s) & mask)[0]
            c, w = r[lab[r] == "correct"], r[lab[r] == t]
            if len(c) and len(w):
                diffs.append(B[w].mean(0) - B[c].mean(0))
                cnt += 1
        return (np.mean(diffs, 0) if diffs else np.full(B.shape[1], np.nan)), cnt

    summary = {}
    for mode in ("raw", "paired"):
        S = {sp: {} for sp in split}
        N = {}
        for sp in split:
            for t in (f"t{k}" for k in TYPES):
                if mode == "raw":
                    ft, fc = freq(sm[sp] & (lab == t)), freq(sm[sp] & (lab == "correct"))
                    S[sp][t] = ft - fc
                    N[(sp, t)] = int((sm[sp] & (lab == t)).sum())
                else:
                    S[sp][t], N[(sp, t)] = paired(sm[sp], t)
        ts = [t for t in (f"t{k}" for k in TYPES) if N[("train", t)] >= (MIN_ROWS if mode == "raw" else 15)]
        print(f"\n=== {mode}: train n per type {({t: N[('train', t)] for t in ts})}  (paired: n = facts)")
        for direction, sg in (("wrong_up", 1), ("correct_up", -1)):
            sc = np.min([sg * S["train"][t] for t in ts], axis=0)
            top = np.argsort(-np.nan_to_num(sc, nan=-9))[: args.top]
            df = pd.DataFrame({"latent": top, "min_train": sc[top]})
            for sp in ("val", "test"):
                df[f"min_{sp}"] = np.min([sg * S[sp][t][top] for t in ts], axis=0)
            for t in ts:
                df[f"test_{t}"] = sg * S["test"][t][top]
            # 단일 latent AUROC: test, wrong(전체) vs correct (활성값 기준)
            tw = sm["test"] & np.isin(lab, [f"t{k}" for k in TYPES])
            tc = sm["test"] & (lab == "correct")
            df["auroc_test"] = [auroc(A[tw][:, j].astype(float), A[tc][:, j].astype(float)) if sg == 1
                                else auroc(A[tc][:, j].astype(float), A[tw][:, j].astype(float)) for j in top]
            df.to_csv(f"{args.out}/{args.position}_{mode}_{direction}.csv", index=False)
            print(f"-- {direction}  (min over {ts})")
            print(df.round(3).head(10).to_string(index=False))
            summary[(mode, direction)] = df

    # 해석용: paired wrong_up 상위 latent의 최대 활성 예시
    top = summary[("paired", "wrong_up")].latent.values[:5]
    print("\n== 해석용: paired/wrong_up 상위 latent의 최대 활성 질문")
    for j in top:
        o = np.argsort(-A[:, j].astype(float))[:5]
        print(f" latent {j}  (active rate: correct {B[lab=='correct'][:, j].mean():.2f} / wrong {B[np.isin(lab, ['t1','t2','t3','t6'])][:, j].mean():.2f})")
        for i in o:
            print(f"    {lab[i]:7s} {rel[i][:10]:10s} {meta[i]['question'][:60]!r} -> {meta[i]['model_answer'][:20]!r}")


if __name__ == "__main__":
    main()
