"""Why did the early experiments (steer_broken.py / ablate_broken.py) look like they worked?   Size of the edit of the early signal vs the early random control.  CPU.
Rebuilds the early broken-side subspace (pooled t-statistic, dev half, K=20) and the early random 20-dim Gaussian subspace (same seed), and measures on the
saved step-0 prompt-end states of the early target rows (test-half broken rows of facts 1/5..4/5) the share of ||h||^2 each projection removes."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "analysis"); torch.set_num_threads(16); import sae_io
D = "data"; K = 20
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); mid = (k >= 1) & (k <= 4)
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r"); enc = sae_io.dream_sae(); Wd = sae_io._load(sae_io.DREAM); Wdec = Wd["W_dec"]; Wu = Wdec / Wdec.norm(dim=1, keepdim=True)
dv = np.where(dev & mid & ((ty == "correct") | (ty == "broken")))[0]; yy = ty[dv] == "correct"
A = torch.cat([enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in dv[b:b + 512]]))) for b in range(0, len(dv), 512)]).numpy()
P, Q_ = A[yy], A[~yy]; tst = (P.mean(0) - Q_.mean(0)) / np.sqrt(P.var(0) / len(P) + Q_.var(0) / len(Q_) + 1e-8); dn = np.argsort(tst)[:K]
Qb = torch.linalg.qr(Wu[dn].T.float())[0]
g_ = torch.Generator().manual_seed(0); _ = torch.randn(Wdec.shape[1], generator=g_); Qr, _ = torch.linalg.qr(torch.randn(Wdec.shape[1], K, generator=g_))
tg = np.where(te & mid & (ty == "broken"))[0]; X = torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in tg]))
sh = lambda Q: ((X @ Q).norm(dim=1) ** 2 / X.norm(dim=1) ** 2).mean().item()
print(f"early target rows {len(tg)}: share of ||h||^2 removed  early broken-signal subspace (t-stat K20) {sh(Qb):.4f} | early random Gaussian 20-dim subspace {sh(Qr):.4f}")
print("activation-weighted: mean firing rate of the early signal latents over all 1/5..4/5 rows used for selection: %.3f" % (A[:, dn] > 0).mean())
