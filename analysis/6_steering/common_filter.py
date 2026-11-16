"""Which "common latent" filter should we use?  Common = fires on many prompts whatever the question and whatever the answer.
Firing rate of every latent over ALL 55,225 rows (layer 20, prompt-end token, step 0).  Latents above the cap are dropped; then the broken-side / entity-side latents are picked on 0/5 facts
by log((rate broken + .01) / (rate entity + .01)) (needs >= 5% of the rows of its own kind), top 20 each.   Caps: 2%, 5%, 10%, 20%, 50% (the paper used 2% on general text).
Reports: latents left, how specific the picked broken-side latents are on facts 1/5..4/5, and within-fact AUROC on facts 1/5..4/5 (not used for picking).
Usage: CUDA_VISIBLE_DEVICES=2 python3 analysis/6_steering/common_filter.py"""
import collections, json, sys
import numpy as np, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "analysis"); import sae_io
D = "data"; K = 20
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, True); H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
N = len(rows); O = np.zeros((N, 16384), bool)
for b in range(0, N, 2048): O[b:b + 2048] = (enc(torch.from_numpy(np.asarray(H[b:b + 2048, 0], dtype=np.float32)).cuda()) > 0).cpu().numpy()
allrate = O.mean(0); print("rows", N, "| latents that ever fire:", int((allrate > 0).sum()), flush=True)
zb, ze = (k == 0) & (ty == "broken"), (k == 0) & (ty == "entity"); rb, re_ = O[zb].mean(0), O[ze].mean(0); lr = np.log((rb + .01) / (re_ + .01))
mi = np.where(mid & np.isin(ty, ["correct", "entity", "broken"]))[0]; Om = O[mi]; tm, fm = ty[mi], fid[mi]
def within(s, a_, b_):
    per = [roc_auc_score((tm[m] == b_).astype(int), s[m]) for u in np.unique(fm) for m in [(fm == u) & np.isin(tm, [a_, b_])] if len(set(tm[m])) == 2]; return np.mean(per)
print(f"{'cap':>5s} | latents left | broken-side picked: ON in 1-4/5 broken / correct / entity (lift) | within-fact AUROC 1-4/5: cor-brk ent-brk cor-ent")
for cap in (.02, .05, .10, .20, .50):
    okb, oke = (allrate <= cap) & (rb >= .05), (allrate <= cap) & (re_ >= .05)
    dn = np.argsort(-np.where(okb, lr, -1e9))[:K]; up = np.argsort(-np.where(oke, -lr, -1e9))[:K]; nb, ne = int(okb.sum()), int(oke.sum())
    b_, c_, e_ = Om[tm == "broken"][:, dn].mean(), Om[tm == "correct"][:, dn].mean(), Om[tm == "entity"][:, dn].mean()
    s = Om[:, dn].sum(1).astype(float) - Om[:, up].sum(1)
    print(f"{cap:>5.2f} | {int((allrate <= cap).sum()):>6d} (brk cand {nb}, ent cand {ne}) | {b_:.3f} / {c_:.3f} / {e_:.3f} ({b_ / max((c_ + e_) / 2, 1e-6):.2f}x) | {within(s, 'correct', 'broken'):.3f} {within(s, 'entity', 'broken'):.3f} {within(s, 'correct', 'entity'):.3f}")
