"""ON/OFF (activation>0) vs STRENGTH readout of SAE latents. Dream, layer 20, prompt end, step 0.  GPU used only for SAE encoding.
Task 1  knows vs does-not-know: facts 5/5 vs 0/5 (rows).            AUROC pooled and inside each relation.
Task 2  same fact, correct vs wrong (facts with 1/5..4/5).          AUROC inside each fact (average over test facts).
Latents (top-K for each side) and everything else are chosen on dev subjects, evaluated on test subjects (50/50 by subject).
on/off score  = #(correct-side latents on) - #(wrong-side latents on),  selected by firing-fraction difference (separation score)
strength score = sum of z-scored activations of the same kind of latents, selected by t-statistic.
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/5_onoff_vs_strength/onoff_vs_strength.py
"""
import json, sys, warnings
import numpy as np, torch
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis")
import sae_io
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
N = len(rows); cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows]); rel = np.array([r["relation"] for r in rows])
fid = np.array([r["sample_id"] for r in rows]); sub = np.array([popqa[r["sample_id"]] for r in rows])
rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
w = {a: b.cuda() for a, b in sae_io._load(sae_io.DREAM).items()}
enc = sae_io._encoder(w, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
H = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")
A = np.zeros((N, 16384), np.float16)
for b in range(0, N, 1024):
    A[b:b + 1024] = enc(torch.from_numpy(np.asarray(H[b:b + 1024, 0], dtype=np.float32)).cuda()).half().cpu().numpy()
print("encoded", A.shape, "mean latents on", float((A > 0).sum(1).mean()), flush=True)

def stats(m_pos, m_neg):
    """dev rows -> on/off separation and t-stat of each latent (pos minus neg)."""
    P, Q = A[m_pos].astype(np.float32), A[m_neg].astype(np.float32)
    sep = (P > 0).mean(0) - (Q > 0).mean(0)
    t = (P.mean(0) - Q.mean(0)) / np.sqrt(P.var(0) / len(P) + Q.var(0) / len(Q) + 1e-8)
    return sep, t, np.concatenate([P, Q]).mean(0), np.concatenate([P, Q]).std(0) + 1e-6

def scores(sep, t, mu, sd, K, idx):
    X = A[idx].astype(np.float32)
    s_on = (X[:, np.argsort(-sep)[:K]] > 0).sum(1) - (X[:, np.argsort(sep)[:K]] > 0).sum(1)
    Z = (X - mu) / sd
    s_st = Z[:, np.argsort(-t)[:K]].sum(1) - Z[:, np.argsort(t)[:K]].sum(1)
    return s_on, s_st

def within(score, y, g, minn=1):
    out, ww = [], []
    for u in np.unique(g):
        m = g == u
        if y[m].min() != y[m].max(): out.append(roc_auc_score(y[m], score[m])); ww.append(1)
    return float(np.mean(out)), len(out)

Ks = (1, 5, 20, 100)
# ---- Task 1: 5/5 (pos) vs 0/5 (neg)
kn, un = (k == 5), (k == 0)
sep, t, mu, sd = stats(dev & kn, dev & un)
print("\n=== Task 1: knows (5/5) vs does not know (0/5). dev", int((dev & kn).sum()), int((dev & un).sum()), "test", int((te & kn).sum()), int((te & un).sum()))
print("  best single latent: separation", round(float(sep.max()), 2), "| -", round(float(-sep.min()), 2))
idx = np.where(te & (kn | un))[0]; y = kn[idx].astype(int)
for K in Ks:
    s_on, s_st = scores(sep, t, mu, sd, K, idx)
    res = []
    for s in (s_on, s_st):
        o, ww = [], []
        for r in np.unique(rel[idx]):
            m = rel[idx] == r
            if y[m].sum() >= 20 and (1 - y[m]).sum() >= 20: o.append(roc_auc_score(y[m], s[m])); ww.append(m.sum())
        res.append((roc_auc_score(y, s), np.average(o, weights=ww)))
    print(f"  K={K:<3d} on/off: pooled {res[0][0]:.3f} within-rel {res[0][1]:.3f} | strength: pooled {res[1][0]:.3f} within-rel {res[1][1]:.3f}")
# ---- Task 2: same fact, correct vs wrong
mid = (k >= 1) & (k <= 4)
sep, t, mu, sd = stats(dev & mid & cor, dev & mid & ~cor)
print("\n=== Task 2: facts 1/5..4/5, correct vs wrong inside the same fact. dev rows", int((dev & mid).sum()), "test rows", int((te & mid).sum()))
print("  best single latent: separation", round(float(sep.max()), 2), "| -", round(float(-sep.min()), 2))
idx = np.where(te & mid)[0]; y = cor[idx].astype(int)
for K in Ks:
    s_on, s_st = scores(sep, t, mu, sd, K, idx)
    a, n1 = within(s_on, y, fid[idx]); b, n2 = within(s_st, y, fid[idx])
    print(f"  K={K:<3d} on/off: within-fact AUROC {a:.3f} | strength: {b:.3f}   (facts {n1})")
