"""Save raw Qwen2.5-7B layer-20 hidden states (pre-SAE) at q_last / prompt_last.

Same rows, prompts and positions as extract_qwen_features.py; output float32 [N, 3584].

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 sae_analysis/extract_qwen_hidden.py --dataset popqa
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
from extract_qwen_features import LAYER, MODEL, REVISION, load_rows, positions  # noqa: E402
from run_inference import build_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["popqa", "triviaqa"])
    ap.add_argument("--output-dir", default="sae_analysis/features")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--all", action="store_true", help="also include the 0-of-5 facts; writes *_all_* files + meta")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    rows = [dict(r, group="kept") for r in load_rows(args.dataset)]
    if args.all:  # kept rows + rows of the 0-of-5 facts, in that order
        from extract_qwen_zero_features import load_zero_rows
        rows += [dict(r, group="zero") for r in load_zero_rows(args.dataset)]
    n, d = len(rows), model.config.hidden_size
    out = {k: np.zeros((n, d), dtype=np.float32) for k in ("q_last", "prompt_last")}
    for start in range(0, n, args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [build_prompt(r["question"]) for r in batch]
        pos = [positions(tok, r["question"], p) for r, p in zip(batch, prompts)]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER + 1]
        ar = torch.arange(len(batch))
        for name, idx in (("q_last", [p[1] for p in pos]), ("prompt_last", [p[2] for p in pos])):
            out[name][start : start + len(batch)] = hs[ar, torch.tensor(idx, device="cuda")].float().cpu().numpy()
        print(f"{start + len(batch)}/{n}", flush=True)

    tag = "_all" if args.all else ""
    for name, arr in out.items():
        np.save(os.path.join(args.output_dir, f"qwen_{args.dataset}{tag}_{name}_hidden.npy"), arr)
    if args.all:
        with open(os.path.join(args.output_dir, f"qwen_{args.dataset}_all_meta.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r.get(k) for k in ("sample_id", "version", "question", "model_answer", "verdict", "reason", "gold_issue", "group")}, ensure_ascii=False) + "\n")
    print(f"done {args.dataset}: {n} rows", flush=True)


if __name__ == "__main__":
    main()
