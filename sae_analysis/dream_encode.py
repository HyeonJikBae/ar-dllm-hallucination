"""Encode saved Dream hidden states with one SAE and write feature files in the Qwen layout.

Output (per prefix = dream_<variant>): <prefix>_<ds>_meta.jsonl + <prefix>_<ds>_<pos>.npy for the kept
facts, and <prefix>_<ds>_zero_meta.jsonl + <prefix>_<ds>_zero_<pos>.npy for the 0-of-5 facts, so the
Qwen analysis scripts run unchanged with MODEL_PREFIX=<prefix> FEAT_DIR=sae_analysis/features_dream_sae.

Usage:
    python3 sae_analysis/dream_encode.py --sae dream-7b_L20_shared-0.2-0.5-0.8_s5_200M
"""

import argparse
import json
import os

import numpy as np
import torch

from dream_sae_compare import DATASETS, FEAT, encode, load_sae

OUT = "sae_analysis/features_dream_sae"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="dream-7b_L20_shared-0.2-0.5-0.8_s5_200M")
    args = ap.parse_args()
    sae, ub = load_sae(args.sae)
    os.makedirs(OUT, exist_ok=True)
    for ds in DATASETS:
        meta = [json.loads(l) for l in open(f"{FEAT}/dream_{ds}_meta.jsonl", encoding="utf-8")]
        zero = np.array([m["group"] == "zero" for m in meta])
        for variant in ("plain", "masked"):
            prefix = f"dream_{variant}"
            for pos in ("q_last", "prompt_last"):
                x = torch.from_numpy(np.load(f"{FEAT}/dream_{ds}_{variant}_{pos}_hidden.npy"))
                with torch.no_grad():
                    acts = torch.cat([encode(sae, x[i:i + 1024], ub) for i in range(0, len(x), 1024)]).numpy().astype(np.float16)
                np.save(f"{OUT}/{prefix}_{ds}_{pos}.npy", acts[~zero])
                np.save(f"{OUT}/{prefix}_{ds}_zero_{pos}.npy", acts[zero])
            for suffix, mask in (("", ~zero), ("_zero", zero)):
                with open(f"{OUT}/{prefix}_{ds}{suffix}_meta.jsonl", "w", encoding="utf-8") as f:
                    for m, keep in zip(meta, mask):
                        if keep:
                            f.write(json.dumps({k: m[k] for k in ("sample_id", "version", "question", "model_answer", "verdict", "reason", "gold_issue")}, ensure_ascii=False) + "\n")
        print(f"{ds}: kept {int((~zero).sum())}, zero {int(zero.sum())}")


if __name__ == "__main__":
    main()
