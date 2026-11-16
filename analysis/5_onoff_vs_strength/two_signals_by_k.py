"""The two signals separately, by k.  KNOW-latents = top-20 latents that fire more on 5/5-correct rows than on 0/5-wrong rows (dev subjects, separation score);
UNKNOWN-latents = top-20 in the opposite direction.  For test rows grouped by fact group k/5 x correct/wrong:
 - share of the 20 KNOW (UNKNOWN) latents that are ON  - mean activation strength of those 20 latents, relative to the 0/5 and 5/5 reference rows (0 = 0/5 wrong median row, 1 = 5/5 correct median row).
Dream, SAE layer 20, prompt-end token.  Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/5_onoff_vs_strength/two_signals_by_k.py"""
import json, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import sae_io
import os
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"; K = int(os.environ.get("TOPK", 20))
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows]); N = len(rows)
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
w = {a: b.cuda() for a, b in sae_io._load(sae_io.DREAM).items()}
enc = sae_io._encoder(w, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
H = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")
for S, name in ((0, "step 0"), (1, "step 8")):
    A = np.zeros((N, 16384), np.float16)
    for b in range(0, N, 1024): A[b:b + 1024] = enc(torch.from_numpy(np.asarray(H[b:b + 1024, S], dtype=np.float32)).cuda()).half().cpu().numpy()
    kn, un = dev & cor & (k == 5), dev & ~cor & (k == 0)
    sep = (A[kn] > 0).mean(0, dtype=np.float32) - (A[un] > 0).mean(0, dtype=np.float32)
    ik, iu = np.argsort(-sep)[:K], np.argsort(sep)[:K]
    print(f"\n===== {name}.   best KNOW latent separation {sep[ik[0]]:.2f}, best UNKNOWN latent {-sep[iu[0]]:.2f}")
    sig = {}
    for nm, ix in (("KNOW", ik), ("UNKNOWN", iu)):
        X = A[:, ix].astype(np.float32); sig[nm] = ((X > 0).mean(1), X.mean(1))
    def rel(v, nm):                                           # 0 = median of 0/5 rows, 1 = median of correct 5/5 rows (test)
        a, b = np.median(v[te & ~cor & (k == 0)]), np.median(v[te & cor & (k == 5)]); return (v - a) / (b - a)
    print("   k/5 correct/wrong |  n  | KNOW: share ON, strength(0..1) | UNKNOWN: share ON, strength(0..1)")
    for g in range(6):
        for c, cn in ((True, "correct"), (False, "wrong  ")):
            m = te & (k == g) & (cor == c)
            if m.sum() < 20: continue
            kv, uv = sig["KNOW"], sig["UNKNOWN"]
            print(f"   {g}/5 {cn}     | {m.sum():>5d} | {kv[0][m].mean():.2f}  {np.median(rel(kv[1], 'k')[m]):5.2f}            | {uv[0][m].mean():.2f}  {np.median(rel(uv[1], 'u')[m]):5.2f}")
