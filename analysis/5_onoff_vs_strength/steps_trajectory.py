"""Does the denoising trajectory help?  Same fact (1/5..4/5), correct vs wrong, within-fact AUROC.  Dream, layer 20, prompt-end token, saved steps 0/8/16/24/31.
Latents/probes are fit on dev subjects and evaluated on test subjects.
 per step s:  SAE on/off (top-K), SAE strength (top-K), raw hidden-state probe
 change:      SAE strength / on-off change between step s and step 0 (top-K)
 baseline:    answer-slot confidence (log-prob, margin; mean/min over 32 slots) at all denoising steps up to s, logistic regression
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/5_onoff_vs_strength/steps_trajectory.py
"""
import json, sys, warnings
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import sae_io
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"; K = 20; SAVE = [0, 8, 16, 24, 31]
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
cor_all = np.array([r["correct"] for r in rows]); k_all = np.array([r["k_correct"] for r in rows])
mid = np.where((k_all >= 1) & (k_all <= 4))[0]
cor = cor_all[mid].astype(int); fid = np.array([rows[i]["sample_id"] for i in mid]); sub = np.array([popqa[rows[i]["sample_id"]] for i in mid])
rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
print("rows", len(mid), "dev", dev.sum(), "test", te.sum(), "correct share", cor.mean().round(3), flush=True)
w = {a: b.cuda() for a, b in sae_io._load(sae_io.DREAM).items()}
enc = sae_io._encoder(w, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
H = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")
RAW = np.stack([np.asarray(H[mid, s], dtype=np.float32) for s in range(5)], 1)       # [n, 5, 3584]
ACT = np.zeros((len(mid), 5, 16384), np.float16)
for s in range(5):
    for b in range(0, len(mid), 1024):
        ACT[b:b + 1024, s] = enc(torch.from_numpy(RAW[b:b + 1024, s]).cuda()).half().cpu().numpy()

def within(score):
    out = [roc_auc_score(cor[te & (fid == u)], score[te & (fid == u)]) for u in np.unique(fid[te]) if cor[te & (fid == u)].min() != cor[te & (fid == u)].max()]
    return float(np.mean(out)), len(out)
def topk_scores(X, on):
    """X [n, 16384] float. on=True: separation score of (X>0) and sum of on-flags; else t-stat and sum of z-scores"""
    P, Q = X[dev & (cor == 1)], X[dev & (cor == 0)]
    if on:
        sep = (P > 0).mean(0) - (Q > 0).mean(0); B = (X > 0).astype(np.float32)
        return B[:, np.argsort(-sep)[:K]].sum(1) - B[:, np.argsort(sep)[:K]].sum(1)
    t = (P.mean(0) - Q.mean(0)) / np.sqrt(P.var(0) / len(P) + Q.var(0) / len(Q) + 1e-8)
    mu, sd = X[dev].mean(0), X[dev].std(0) + 1e-6; Z = (X - mu) / sd
    return Z[:, np.argsort(-t)[:K]].sum(1) - Z[:, np.argsort(t)[:K]].sum(1)
def probe(X):
    sc = StandardScaler().fit(X[dev]); m = LogisticRegression(C=0.001, max_iter=300).fit(sc.transform(X[dev]), cor[dev])
    return m.decision_function(sc.transform(X))

print(f"\nTop-K = {K}. Within-fact AUROC (test facts).   chance 0.5")
print("step  SAE on/off  SAE strength  raw probe |  change vs step0: on/off  strength")
for s in range(5):
    X = ACT[:, s].astype(np.float32)
    a, nf = within(topk_scores(X, True)); b, _ = within(topk_scores(X, False)); c, _ = within(probe(RAW[:, s]))
    if s:
        dX = X - ACT[:, 0].astype(np.float32); dOn = (X > 0).astype(np.float32) - (ACT[:, 0] > 0).astype(np.float32)
        d1, _ = within(topk_scores(dOn, False)); d2, _ = within(topk_scores(dX, False))
        print(f"{SAVE[s]:>3d}   {a:.3f}       {b:.3f}         {c:.3f}    |   {d1:.3f}              {d2:.3f}")
    else:
        print(f"{SAVE[s]:>3d}   {a:.3f}       {b:.3f}         {c:.3f}    |   -")
print("   (facts:", nf, ")")
# trajectory of raw hidden state: concatenation of steps 0..s
for s in (1, 2, 3, 4):
    c, _ = within(probe(RAW[:, :s + 1].reshape(len(mid), -1)))
    print(f"raw probe on steps 0..{SAVE[s]} concatenated: {c:.3f}")
# confidence baseline
LP = np.load(f"{F}/conf_lp.npy", mmap_mode="r")[mid].astype(np.float32); MG = np.load(f"{F}/conf_mg.npy", mmap_mode="r")[mid].astype(np.float32)
f = np.concatenate([LP.mean(2), LP.min(2), MG.mean(2), MG.min(2)], 1)      # [n, 4*32]  (per denoising step 0..31)
def cols(smax): return np.concatenate([f[:, j * 32: j * 32 + smax + 1] for j in range(4)], 1)
print("\nconfidence baseline (log-prob/margin of all 32 slots, all steps up to s):")
for s in (0, 8, 16, 24, 31):
    X = cols(s); sc = StandardScaler().fit(X[dev]); m = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(X[dev]), cor[dev])
    print(f"  up to step {s:>2d}: {within(m.decision_function(sc.transform(X)))[0]:.3f}")
