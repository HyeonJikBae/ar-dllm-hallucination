"""Which (input variant, SAE) fits Dream best? Encode saved hidden states with all 4 Dream SAEs.

For each (variant in plain/masked) x (position) x (SAE) report, per dataset:
  recon cosine (mean), L0 (mean active latents), and K-vs-U AUROC of a diff-in-means know-axis on
  binary latent activity (60/40 split by fact, mean over splits). K = correct rows, U = 0-of-5 rows.

Usage:
    python3 sae_analysis/dream_sae_compare.py
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from three_group import auroc  # noqa: E402

FEAT = "sae_analysis/features_dream"
SAE_ROOT = "sae/dream_pilot_200M"
SAES = ["dream-7b_L20_t0.2_s1_200M", "dream-7b_L20_t0.5_s2_200M", "dream-7b_L20_t0.8_s3_200M",
        "dream-7b_L20_shared-0.2-0.5-0.8_s5_200M"]
DATASETS = ["popqa", "triviaqa"]
torch.set_num_threads(16)


def load_sae(name):
    d = f"{SAE_ROOT}/{name}"
    cfg = json.load(open(f"{d}/cfg.json"))
    w = load_file(f"{d}/sae_weights.safetensors")
    return {k: w[k].float() for k in ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")}, cfg.get("apply_b_dec_to_input", True)


@torch.no_grad()
def encode(sae, x, use_b_dec):
    pre = ((x - sae["b_dec"]) if use_b_dec else x) @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


def main():
    meta, labels = {}, {}
    for ds in DATASETS:
        m = [json.loads(l) for l in open(f"{FEAT}/dream_{ds}_meta.jsonl", encoding="utf-8")]
        meta[ds] = m
        g = np.array(["K" if (r["group"] == "kept" and r["verdict"] == "correct") else
                      ("U" if r["group"] == "zero" else "W") for r in m])
        g[[r["gold_issue"] for r in m]] = "X"
        labels[ds] = (np.array([r["sample_id"] for r in m]), g)

    results = []
    for name in SAES:
        sae, ub = load_sae(name)
        for variant in ("plain", "masked"):
            for pos in ("q_last", "prompt_last"):
                row = {"sae": name.split("_")[2] if "shared" not in name else "shared", "variant": variant, "pos": pos}
                for ds in DATASETS:
                    x = torch.from_numpy(np.load(f"{FEAT}/dream_{ds}_{variant}_{pos}_hidden.npy"))
                    acts = torch.cat([encode(sae, x[i:i + 1024], ub) for i in range(0, len(x), 1024)])
                    rec = torch.cat([acts[i:i + 1024] @ sae["W_dec"] + sae["b_dec"] for i in range(0, len(x), 1024)])
                    cos = torch.cosine_similarity(rec, x).mean().item()
                    l0 = (acts > 0).float().sum(1).mean().item()
                    B = (acts > 0).numpy().astype(np.float32)
                    sid, g = labels[ds]
                    keep = np.isin(g, ["K", "U"])
                    facts = np.array(sorted(set(sid[keep])))
                    aus = []
                    for seed in range(10):
                        rng = np.random.default_rng(seed)
                        tr = np.isin(sid, rng.permutation(facts)[: int(0.6 * len(facts))])
                        d = B[tr & (g == "K")].mean(0) - B[tr & (g == "U")].mean(0)
                        s = B @ d
                        aus.append(auroc(s[~tr & (g == "K")], s[~tr & (g == "U")]))
                    row[f"{ds}_cos"], row[f"{ds}_l0"], row[f"{ds}_auc"] = cos, l0, float(np.mean(aus))
                results.append(row)
                print(f"{row['sae']:7s} {variant:6s} {pos:11s} | popqa cos {row['popqa_cos']:.3f} l0 {row['popqa_l0']:5.1f} AUC {row['popqa_auc']:.3f}"
                      f" | trivia cos {row['triviaqa_cos']:.3f} l0 {row['triviaqa_l0']:5.1f} AUC {row['triviaqa_auc']:.3f}", flush=True)
    json.dump(results, open("sae_analysis/dream_sae_compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
