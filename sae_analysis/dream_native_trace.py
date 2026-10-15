"""Dream-native confidence: re-decode the 20,000 extension rows (topk_margin, 32 steps, same as dream_topk_extend.py) and record, at every
denoising step, the top-1 log-prob and the top1-top2 margin of each of the 32 canvas positions, plus which positions are still masked.
From this: confidence of the tokens that were actually committed, and canvas-wide step-0 confidence (dream_native_features.py).
Chunked + resumable: sae_analysis/results_twelve/dream_native/chunk_XXXX.npz (640 rows each).
Usage: CUDA_VISIBLE_DEVICES=0 /home/minyeol/.conda/envs/sae-dream/bin/python sae_analysis/dream_native_trace.py [--limit N]
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "sae")):
    sys.path.insert(0, p)
OUTD = "sae_analysis/results_twelve/dream_native"; ROWS = "sae_analysis/results_twelve/dream_ext/rows.jsonl"
CANVAS, STEPS, BATCH, CHUNK = 32, 32, 64, 640

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
    import torch
    from dream_acts import load_dream
    from run_inference import build_prompt, extract_answer
    os.makedirs(OUTD, exist_ok=True)
    rows = [json.loads(l) for l in open(ROWS, encoding="utf-8")]
    if a.limit: rows = rows[:a.limit]
    model, tok = load_dream("cuda"); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    for c0 in range(0, len(rows), CHUNK):
        path = f"{OUTD}/chunk_{c0:05d}.npz"
        if os.path.exists(path): continue
        part = rows[c0:c0 + CHUNK]; LP, MG, MK, FX, ANS = [], [], [], [], []
        for b in range(0, len(part), BATCH):
            batch = part[b:b + BATCH]
            enc = tok([build_prompt(r["question"]) for r in batch], return_tensors="pt", padding=True, truncation=True).to("cuda")
            lp = torch.zeros(len(batch), STEPS, CANVAS, device="cuda"); mg = torch.zeros_like(lp); mk = torch.zeros(len(batch), STEPS, CANVAS, dtype=torch.bool, device="cuda")
            def hook(step, x, logits):
                z = logits[:, -CANVAS:].float(); lse = torch.logsumexp(z, -1); t2 = z.topk(2, -1).values
                lp[:, step] = t2[..., 0] - lse; mg[:, step] = t2[..., 0] - t2[..., 1]; mk[:, step] = x[:, -CANVAS:] == tok.mask_token_id
                return logits
            with torch.inference_mode():
                out = model.diffusion_generate(inputs=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=CANVAS, steps=STEPS,
                    temperature=0.0, alg="topk_margin", alg_temp=0.0, mask_token_id=tok.mask_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    generation_logits_hook_func=hook)
            LP.append(lp.cpu().numpy().astype(np.float16)); MG.append(mg.cpu().numpy().astype(np.float16)); MK.append(mk.cpu().numpy())
            FX.append(out[:, -CANVAS:].cpu().numpy())
            ANS += [extract_answer(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)) for o in out]
        np.savez(path, lp=np.concatenate(LP), mg=np.concatenate(MG), mk=np.concatenate(MK), fx=np.concatenate(FX), ans=np.array(ANS, dtype=object), eos=tok.eos_token_id, allow_pickle=True)
        print(f"chunk {c0} saved", flush=True)
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
