"""SAE features for the 0-of-5 facts (rows excluded from "6. excl 0of5") = "doesn't know" group.

Same prompts, positions and SAE as extract_qwen_features.py; rows come from
"5. reviewed 5" minus the sample_ids kept in "6. excl 0of5".

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 sae_analysis/extract_qwen_zero_features.py --dataset popqa
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
from extract_qwen_features import (  # noqa: E402
    GRADED, LAYER, MODEL, REVISION, SAE_DIR, JumpReLUSAE, positions,
)
from run_inference import build_prompt  # noqa: E402

FULL = "analysis/graded/5. reviewed 5/qwen2.5-7b_{dataset}_reclassified.jsonl"


def load_zero_rows(dataset):
    kept = {json.loads(l)["sample_id"] for l in open(GRADED.format(dataset=dataset), encoding="utf-8")}
    rows = []
    with open(FULL.format(dataset=dataset), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["sample_id"] not in kept and d["verdict"] != "uncertain":
                rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["popqa", "triviaqa"])
    ap.add_argument("--output-dir", default="sae_analysis/features")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    sae = JumpReLUSAE(SAE_DIR, "cuda")

    rows = load_zero_rows(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    n, d_sae = len(rows), sae.W_enc.shape[1]
    out = {k: np.zeros((n, d_sae), dtype=np.float16) for k in ("q_last", "prompt_last")}

    for start in range(0, n, args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [build_prompt(r["question"]) for r in batch]
        pos = [positions(tok, r["question"], p) for r, p in zip(batch, prompts)]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER + 1]
        ar = torch.arange(len(batch))
        for name, idx in (("q_last", [p[1] for p in pos]), ("prompt_last", [p[2] for p in pos])):
            x = hs[ar, torch.tensor(idx, device="cuda")]
            out[name][start : start + len(batch)] = sae.encode(x).cpu().numpy().astype(np.float16)
        if (start // args.batch_size) % 20 == 0:
            print(f"{start + len(batch)}/{n}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for name, arr in out.items():
        np.save(os.path.join(args.output_dir, f"qwen_{args.dataset}_zero_{name}.npy"), arr)
    with open(os.path.join(args.output_dir, f"qwen_{args.dataset}_zero_meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("sample_id", "version", "question", "model_answer", "verdict", "reason", "gold_issue")}, ensure_ascii=False) + "\n")
    print(f"done {args.dataset}: {n} rows")


if __name__ == "__main__":
    main()
