"""Steering directions (layer 20, last prompt position) for Dream, fitted per subject fold on the MIXED facts of the other folds.
  dim     mean over facts of (mean hidden of right wordings - mean hidden of wrong wordings)   [raw layer-20 state at prompt_last]
  sae     sum over the 20 SAE latents with the largest paired t (as in signals_unified) of sign(t) * decoder vector
  random  fixed random direction (control)
Each direction is a unit vector; sigma = std of its projection over the training rows (steering strength is alpha * sigma).
Also reports a sanity check: held-out within-fact AUROC of the projection (does the direction separate right from wrong wordings?).
Facts for the steering run: 1,000 random facts out of the 4,000 (all folds), saved as subset.
Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=sae_analysis:grounded_dream /home/minyeol/.conda/envs/sae/bin/python sae_analysis/steer_directions.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "sae_analysis"); sys.path.insert(0, "grounded_dream")
from signals_unified import DREAM_SAE, load_sae, within_auc  # noqa: E402
F = "sae_analysis/features_dream_step0_ext/"; OUT = "sae_analysis/results_twelve/steer"
meta = [json.loads(l) for l in open(F + "meta.jsonl", encoding="utf-8")]
hl = np.load(F + "hid_last.npy").astype(np.float32)
sid = np.array([m["sample_id"] for m in meta]); y = np.array([m["contain"] for m in meta]).astype(int)
subj = {}
for l in open("data/popqa/1. popqa_raw.jsonl", encoding="utf-8"):
    x = json.loads(l); subj[str(x["id"])] = x["s_uri"].rsplit("/", 1)[-1]
facts = np.array(sorted(set(sid))); fidx = {f: i for i, f in enumerate(facts)}; rf = np.array([fidx[f] for f in sid])
ysum = np.bincount(rf, weights=y, minlength=len(facts)); paired = (ysum > 0) & (ysum < 5)
subjects = np.unique([subj[f] for f in facts]); fos = {s: i % 5 for i, s in enumerate(np.random.default_rng(0).permutation(subjects))}
ffold = np.array([fos[subj[f]] for f in facts]); rfold = ffold[rf]
from sae_lens import SAE  # noqa: E402
sae = SAE.load_from_disk(DREAM_SAE, device="cpu").eval(); Wdec = sae.W_dec.detach().float().numpy()
pr = np.where(paired[rf])[0]
A = load_sae("dream")(hl[pr]); Ap = np.zeros((len(facts), A.shape[1]))  # paired-fact diffs in latent space
Mc, Mw = np.zeros((len(facts), A.shape[1])), np.zeros((len(facts), A.shape[1])); Hc, Hw = np.zeros((len(facts), 3584)), np.zeros((len(facts), 3584))
rr = rf[pr]; yy = y[pr]
nc, nw = np.bincount(rr[yy == 1], minlength=len(facts)), np.bincount(rr[yy == 0], minlength=len(facts))
np.add.at(Mc, rr[yy == 1], A[yy == 1]); np.add.at(Mw, rr[yy == 0], A[yy == 0]); np.add.at(Hc, rr[yy == 1], hl[pr][yy == 1]); np.add.at(Hw, rr[yy == 0], hl[pr][yy == 0])
Dm = Mc / np.maximum(nc, 1)[:, None] - Mw / np.maximum(nw, 1)[:, None]; Dh = Hc / np.maximum(nc, 1)[:, None] - Hw / np.maximum(nw, 1)[:, None]
rng = np.random.default_rng(123)
dirs = np.zeros((5, 3, 3584), np.float32); sig = np.zeros((5, 3), np.float32)
for k in range(5):
    trf = paired & (ffold != k); trows = np.where(np.isin(rf, np.where(trf)[0]))[0]
    d_dim = Dh[trf].mean(0)
    Dt = Dm[trf]; t = Dt.mean(0) / (Dt.std(0, ddof=1) / np.sqrt(len(Dt)) + 1e-9)
    o = np.argsort(-np.abs(t))[:20]; d_sae = (np.sign(t[o])[:, None] * Wdec[o]).sum(0)
    d_rnd = rng.standard_normal(3584)
    for j, d in enumerate((d_dim, d_sae, d_rnd)):
        u = d / np.linalg.norm(d); dirs[k, j] = u; sig[k, j] = (hl[trows] @ u).std()
    te = np.where((rfold == k) & paired[rf])[0]
    au = []
    for j in range(3):
        s_ = hl[te] @ dirs[k, j]; fs = np.unique(rf[te])
        au.append(np.nanmean([within_auc(s_[rf[te] == f], y[te][rf[te] == f]) for f in fs]))
    print(f"fold {k}: train facts {trf.sum()}, test facts {len(np.unique(rf[te]))}; 보류 폴드 같은 사실 안 AUROC dim {au[0]:.3f} sae {au[1]:.3f} random {au[2]:.3f}; sigma {sig[k].round(1)}", flush=True)
sub = np.random.default_rng(7).permutation(facts)[:1000]
os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/directions.npz", dirs=dirs, sigma=sig, subset=sub, facts=facts, ffold=ffold)
print("saved; subset facts", len(sub), "of which mixed", int(paired[[fidx[f] for f in sub]].sum()))
sys.stdout.flush(); os._exit(0)
