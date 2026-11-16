"""Paper-style single latents.  KNOW latent = argmax over latents of the MIN over wrong-type groups t in {entity, broken} of separation(5/5-correct rows vs 0/5-wrong rows of type t)  (MaxMin);
UNKNOWN latent = the same in the opposite direction.  Picked on dev subjects.  Then on TEST rows: share of rows where that one latent is ON, by fact group k/5 x correct/wrong.  Dream, SAE layer 20, prompt-end token.
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/5_onoff_vs_strength/paper_style_top1.py"""
import json, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import sae_io
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows]); N = len(rows)
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows])
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
w = {a: b.cuda() for a, b in sae_io._load(sae_io.DREAM).items()}
enc = sae_io._encoder(w, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
H = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")
know = cor & (k == 5); TYPES = ("entity", "broken")
for S, name in ((0, "step 0"), (1, "step 8")):
    A = np.zeros((N, 16384), np.float16)
    for b in range(0, N, 1024): A[b:b + 1024] = enc(torch.from_numpy(np.asarray(H[b:b + 1024, S], dtype=np.float32)).cuda()).half().cpu().numpy()
    on = lambda m: (A[m] > 0).mean(0, dtype=np.float32)
    seps_dev = np.stack([on(dev & know) - on(dev & ~cor & (k == 0) & (ty == t)) for t in TYPES])
    seps_te = np.stack([on(te & know) - on(te & ~cor & (k == 0) & (ty == t)) for t in TYPES])
    jk, ju = int(np.argmax(seps_dev.min(0))), int(np.argmax((-seps_dev).min(0)))
    print(f"\n===== {name}")
    print(f"KNOW latent {jk}: MaxMin dev {seps_dev.min(0)[jk]:.2f} | test separation entity {seps_te[0, jk]:.2f}, broken {seps_te[1, jk]:.2f}")
    print(f"UNKNOWN latent {ju}: MaxMin dev {(-seps_dev).min(0)[ju]:.2f} | test separation entity {-seps_te[0, ju]:.2f}, broken {-seps_te[1, ju]:.2f}")
    for nm, j in (("KNOW", jk), ("UNKNOWN", ju)):
        v = (A[:, j] > 0)
        for cn, c in (("correct rows", True), ("wrong rows", False)):
            print(f"\n[{nm} latent {j}, {cn}, {name}]  k  | n | share ON")
            for g in range(6):
                m = te & (k == g) & (cor == c)
                if m.sum() >= 20: print(f"      {g}/5 | {m.sum():>6d} | {v[m].mean():.2f}")
