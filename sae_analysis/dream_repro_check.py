"""Which diffusion settings reproduce the saved Dream answers?

Re-generates the first N rows of analysis/inference/dream-7b_<dataset>.jsonl (file order, as the
original run_inference.py did) with max_new_tokens=16 and several step counts / batch sizes, and
reports the exact-match rate against the saved model_answer.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 sae_analysis/dream_repro_check.py --dataset popqa --n 64
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_inference import build_prompt, generate_batch  # noqa: E402

MODEL = "Dream-org/Dream-v0-Base-7B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="popqa")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--steps", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 1])
    args = ap.parse_args()

    rows = []
    for line in open(f"analysis/inference/dream-7b_{args.dataset}.jsonl", encoding="utf-8"):
        rows.append(json.loads(line))
        if len(rows) >= args.n:
            break
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

    prompts = [build_prompt(r["question"]) for r in rows]
    saved = [r["model_answer"] for r in rows]
    for bs in args.batch_sizes:
        for steps in args.steps:
            out = []
            for i in range(0, len(prompts), bs):
                out += generate_batch(model, tok, prompts[i:i + bs], max_new_tokens=args.max_new_tokens, diffusion_steps=steps)
            match = sum(a == b for a, b in zip(out, saved))
            print(f"batch {bs:2d} steps {steps:2d} max_new_tokens {args.max_new_tokens}: exact match {match}/{len(rows)}", flush=True)
            if match < len(rows):
                bad = [(s, o) for s, o in zip(saved, out) if s != o][:2]
                print("     e.g. saved vs new:", bad, flush=True)


if __name__ == "__main__":
    main()
