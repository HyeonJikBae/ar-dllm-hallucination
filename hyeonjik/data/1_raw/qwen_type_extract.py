"""Qwen2.5-7B, same rows and same two signals as dream_conf_vs_sae.py: first-answer-token confidence and the layer-20 state at prompt_last.

Rows: see select().
Per row (prompt = run_inference.build_prompt(question), no answer):
  conf_lp  top-1 log-prob of the next token at the last prompt position
  conf_mg  top-1 minus top-2 logit
  hid      layer-20 residual (output of decoder layer 20 = hidden_states[21]) at the last prompt token, float16 [N, 3584]
Outputs: hyeonjik/data/1_raw/qwen_hidden/{rows.json, conf_lp.npy, conf_mg.npy, hid.npy}
Usage:   CUDA_VISIBLE_DEVICES=0 python3 sae_analysis/qwen_conf_extract.py [--limit N] [--batch-size 8]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_inference import build_prompt  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B"
REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
OUT = "hyeonjik/data/1_raw/qwen_hidden"


def select():
    """(a) wrong rows typed entity/broken (type analysis), (b) rows of 1-4/5 facts with both a correct and a wrong row (within-fact analysis)."""
    rows = [json.loads(l) for l in open("hyeonjik/data/3_classification/full/qwen_final_full.jsonl", encoding="utf-8")]
    isc = lambda r: bool(r["correct"])
    a = {i for i, r in enumerate(rows) if r["analysis_use"] and not r["correct"] and r["type3"] in ("entity", "broken")}
    ok = [i for i, r in enumerate(rows) if r["analysis_use"] and r["fact_group"] in ("1/5", "2/5", "3/5", "4/5")]
    byf = collections.defaultdict(list)
    for i in ok:
        byf[rows[i]["sample_id"]].append(i)
    b = {i for f, ix in byf.items() if any(isc(rows[j]) for j in ix) and any(not isc(rows[j]) for j in ix) for i in ix}
    return rows, sorted(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    rows, keep = select()
    if args.limit:
        keep = keep[: args.limit]
    os.makedirs(OUT, exist_ok=True)
    json.dump(keep, open(f"{OUT}/rows.json", "w"))
    print("rows", len(keep), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16).cuda().eval()
    cap = {}
    model.model.layers[20].register_forward_hook(lambda m, a, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1].detach()))
    n = len(keep)
    lp, mg = np.zeros(n, np.float32), np.zeros(n, np.float32)
    hid = np.zeros((n, 3584), np.float16)
    order = sorted(range(n), key=lambda j: len(rows[keep[j]]["question"]))
    with torch.inference_mode():
        for b in range(0, n, args.batch_size):
            idx = order[b : b + args.batch_size]
            enc = tok([build_prompt(rows[keep[j]]["question"]) for j in idx], return_tensors="pt", padding=True).to("cuda")
            out = model.model(**enc)
            logits = model.lm_head(out.last_hidden_state[:, -1]).float()
            lse = torch.logsumexp(logits, -1)
            t2 = logits.topk(2, -1).values
            lp[idx] = (t2[:, 0] - lse).cpu().numpy()
            mg[idx] = (t2[:, 0] - t2[:, 1]).cpu().numpy()
            hid[idx] = cap["h"].to(torch.float16).cpu().numpy()
            if (b // args.batch_size) % 100 == 0:
                print(f"{b}/{n}", flush=True)
    np.save(f"{OUT}/conf_lp.npy", lp)
    np.save(f"{OUT}/conf_mg.npy", mg)
    np.save(f"{OUT}/hid.npy", hid)
    print("done", n, flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
