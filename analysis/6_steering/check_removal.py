"""Does projecting out the span of the 20 decoder directions really switch the 20 broken-signal latents off?  (CPU)  Uses the all-rows within-fact MaxMin broken signal (as in crossfit_broken.py --all) on the saved step-0 prompt-end states of facts 1/5..4/5.
Reports, before / after the projection, for the 20 broken-side latents: share of (row, latent) pairs ON and mean activation; and the same for 20 random latents of the same firing-rate deciles, plus how much of ||h|| is removed."""
import collections, json, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis"); torch.set_num_threads(16)
import sae_io
D = "data"; K = 20
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
idx = np.where(mid & np.isin(ty, ["correct", "broken", "entity"]))[0]; H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
X = torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in idx])); enc = sae_io.dream_sae(); A = torch.cat([enc(X[b:b + 1024]) for b in range(0, len(X), 1024)]).numpy(); on = (A > 0).astype(np.float32)
t = ty[idx]; f = fid[idx]; kk = k[idx]; byf = collections.defaultdict(lambda: {"i": collections.defaultdict(list), "k": 0})
for j, (ff, tt, k_) in enumerate(zip(f, t, kk)): byf[ff]["i"][tt].append(j); byf[ff]["k"] = k_
rate = lambda ff, tt: on[byf[ff]["i"][tt]].mean(0); d = {g: [] for g in (1, 2, 3, 4)}; e = []
for ff, v in byf.items():
    if v["i"]["correct"] and v["i"]["broken"]: d[v["k"]].append(rate(ff, "correct") - rate(ff, "broken"))
    if v["i"]["entity"] and v["i"]["broken"]: e.append(rate(ff, "entity") - rate(ff, "broken"))
sc_b = np.min([-np.mean(d[g], 0) for g in d if d[g]] + [-np.mean(e, 0)], 0); allrate = on.mean(0); okb = (allrate <= .5) & (on[t == "broken"].mean(0) >= .05)
dn = np.argsort(-np.where(okb, sc_b, -9))[:K]
Wd = sae_io._load(sae_io.DREAM)["W_dec"]; Wu = Wd / Wd.norm(dim=1, keepdim=True); Q = torch.linalg.qr(Wu[dn].T.float())[0]
cand = np.where(allrate > .01)[0]; qs = np.quantile(allrate[cand], np.linspace(0, 1, 11)); dec = np.clip(np.digitize(allrate, qs[1:-1]), 0, 9); rr = np.random.default_rng(5); rl = []
for j in dn: rl.append(int(rr.choice([c for c in cand if dec[c] == dec[j] and c not in set(dn) and c not in rl])))
Xp = X - (X @ Q) @ Q.T; Ap = torch.cat([enc(Xp[b:b + 1024]) for b in range(0, len(Xp), 1024)]).numpy()
print(f"rows {len(idx)}; ||h|| mean {X.norm(dim=1).mean():.1f}; share of ||h||^2 removed by the projection {(1 - (Xp.norm(dim=1) ** 2 / X.norm(dim=1) ** 2)).mean():.3f}")
for nm, L in (("20 broken-side latents (the removed ones)", dn), ("20 matched random latents", rl)):
    print(f"{nm}: ON before {(A[:, L] > 0).mean():.3f} -> after {(Ap[:, L] > 0).mean():.3f} | mean activation before {A[:, L].mean():.2f} -> after {Ap[:, L].mean():.2f}")
b = (t == "broken"); print(f"broken rows only, broken-side latents: ON {(A[b][:, dn] > 0).mean():.3f} -> {(Ap[b][:, dn] > 0).mean():.3f}; mean activation {A[b][:, dn].mean():.2f} -> {Ap[b][:, dn].mean():.2f}")
