"""Extract Qwen2.5-7B layer-20 SAE feature activations at question-final positions.

For every graded (sample_id, version) row, run the same zero-shot prompt used
in run_inference.py (question only, no model answer), take the residual stream
after decoder layer 20 (hidden_states[21]) and encode it with the BatchTopK/
JumpReLU SAE. Two positions are stored per row:

  q_last      last token of the question text (e.g. "?")
  prompt_last last token of the prompt ("Answer:" -> ":")

Outputs (dense float16, [N, d_sae]) plus a metadata jsonl aligned by row index.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 sae_analysis/extract_qwen_features.py \
        --dataset popqa --output-dir sae_analysis/features
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_inference import build_prompt  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B"
REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
LAYER = 20
SAE_DIR = "sae/qwen2.5-7b_L20_batchtopk_16k_k100_20M"
GRADED = "analysis/graded/6. excl 0of5/qwen2.5-7b_{dataset}_reclassified.jsonl"


class JumpReLUSAE:
    def __init__(self, sae_dir, device):
        w = load_file(os.path.join(sae_dir, "sae_weights.safetensors"))
        self.W_enc = w["W_enc"].to(device)
        self.W_dec = w["W_dec"].to(device)
        self.b_enc = w["b_enc"].to(device)
        self.b_dec = w["b_dec"].to(device)
        self.threshold = w["threshold"].to(device)

    @torch.no_grad()
    def encode(self, x):
        pre = (x.float() - self.b_dec) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    @torch.no_grad()
    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec


def load_rows(dataset):
    rows = []
    with open(GRADED.format(dataset=dataset), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["verdict"] == "uncertain":
                continue
            rows.append(d)
    return rows


def positions(tokenizer, question, prompt):
    enc = tokenizer(prompt, return_offsets_mapping=True)
    q_end = prompt.index(question) + len(question)
    q_last = max(i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < q_end)
    return enc["input_ids"], q_last, len(enc["input_ids"]) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["popqa", "triviaqa"])
    ap.add_argument("--output-dir", default="sae_analysis/features")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16
    ).to(device).eval()
    sae = JumpReLUSAE(SAE_DIR, device)

    rows = load_rows(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    n, d_sae = len(rows), sae.W_enc.shape[1]
    out = {k: np.zeros((n, d_sae), dtype=np.float16) for k in ("q_last", "prompt_last")}
    cos = {k: [] for k in out}

    for start in range(0, n, args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [build_prompt(r["question"]) for r in batch]
        pos = [positions(tok, r["question"], p) for r, p in zip(batch, prompts)]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER + 1]
        for name, idx in (("q_last", [p[1] for p in pos]), ("prompt_last", [p[2] for p in pos])):
            x = hs[torch.arange(len(batch)), torch.tensor(idx, device=device)]
            acts = sae.encode(x)
            cos[name].append(torch.cosine_similarity(sae.decode(acts), x.float()).cpu())
            out[name][start : start + len(batch)] = acts.cpu().numpy().astype(np.float16)
        print(f"{start + len(batch)}/{n}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for name, arr in out.items():
        np.save(os.path.join(args.output_dir, f"qwen_{args.dataset}_{name}.npy"), arr)
    with open(os.path.join(args.output_dir, f"qwen_{args.dataset}_meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("sample_id", "version", "question", "model_answer", "verdict", "reason", "gold_issue")}, ensure_ascii=False) + "\n")
    for name, c in cos.items():
        c = torch.cat(c)
        print(f"{name}: recon cosine mean={c.mean():.3f} min={c.min():.3f}; l0 mean={(out[name] != 0).sum(1).mean():.1f}")


if __name__ == "__main__":
    main()
