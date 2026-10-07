"""Qwen2.5-7B layer-20 SAE features for the final PopQA labels (7. final).

Same capture rule as extract_qwen_features.py (residual after decoder layer 20, JumpReLU SAE,
positions q_last / prompt_last) but over all 11,045 facts x 5 versions.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 sae_analysis/extract_features_refined.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_qwen_features import MODEL, REVISION, LAYER, SAE_DIR, JumpReLUSAE, positions  # noqa: E402
from run_inference import build_prompt  # noqa: E402

from build_meta_from_final import legacy_rows, KEYS  # noqa: E402  (rows come from 7. final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="sae_analysis/features_refined")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = legacy_rows()          # 'uncertain' rows are already dropped
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16).cuda().eval()
    sae = JumpReLUSAE(SAE_DIR, "cuda")

    n, d_sae = len(rows), sae.W_enc.shape[1]
    out = {k: np.zeros((n, d_sae), dtype=np.float16) for k in ("q_last", "prompt_last")}
    cos = {k: [] for k in out}
    for start in range(0, n, args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [build_prompt(r["question"]) for r in batch]
        pos = [positions(tok, r["question"], p) for r, p in zip(batch, prompts)]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER + 1]
        for name, idx in (("q_last", [p[1] for p in pos]), ("prompt_last", [p[2] for p in pos])):
            x = hs[torch.arange(len(batch)), torch.tensor(idx, device="cuda")]
            acts = sae.encode(x)
            cos[name].append(torch.cosine_similarity(sae.decode(acts), x.float()).cpu())
            out[name][start : start + len(batch)] = acts.cpu().numpy().astype(np.float16)
        if (start // args.batch_size) % 50 == 0:
            print(f"{start + len(batch)}/{n}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for name, arr in out.items():
        np.save(os.path.join(args.output_dir, f"qwen_popqa_{name}.npy"), arr)
    keys = KEYS
    with open(os.path.join(args.output_dir, "qwen_popqa_meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in keys}, ensure_ascii=False) + "\n")
    for name, c in cos.items():
        c = torch.cat(c)
        print(f"{name}: recon cosine mean={c.mean():.3f} min={c.min():.3f}; l0 mean={(out[name] != 0).sum(1).mean():.1f}")


if __name__ == "__main__":
    main()
