"""Gap score = (share of KNOW latents ON) - (share of UNKNOWN latents ON), top-20 each (picked on dev: 5/5-correct vs 0/5-wrong).  Dream, SAE, prompt end.
Question: among WRONG rows, does a high gap mean "room for improvement" = the model answers this fact correctly under other wordings (k>=1)?
Reports on test wrong rows: mean gap by k, AUROC(gap: k>=1 vs k=0) pooled and inside each relation, Spearman(gap, k), and the same for the relation-only baseline.
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/5_onoff_vs_strength/gap_score.py"""
import json, sys, warnings
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import sae_io
import os
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"; K = int(os.environ.get("TOPK", 20))
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows]); rel = np.array([r["relation"] for r in rows]); N = len(rows)
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
w = {a: b.cuda() for a, b in sae_io._load(sae_io.DREAM).items()}
enc = sae_io._encoder(w, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
H = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")
wr = te & ~cor
for S, name in ((0, "step 0"), (1, "step 8")):
    A = np.zeros((N, 16384), np.float16)
    for b in range(0, N, 1024): A[b:b + 1024] = enc(torch.from_numpy(np.asarray(H[b:b + 1024, S], dtype=np.float32)).cuda()).half().cpu().numpy()
    sep = (A[dev & cor & (k == 5)] > 0).mean(0, dtype=np.float32) - (A[dev & ~cor & (k == 0)] > 0).mean(0, dtype=np.float32)
    ik, iu = np.argsort(-sep)[:K], np.argsort(sep)[:K]
    gap = (A[:, ik] > 0).mean(1) - (A[:, iu] > 0).mean(1)
    print(f"\n===== {name}: gap score on test WRONG rows")
    for g in range(5): m = wr & (k == g); print(f"  fact {g}/5 wrong rows  n={m.sum():>6d}  mean gap {gap[m].mean():+.2f}")
    y = (k[wr] >= 1).astype(int); a = roc_auc_score(y, gap[wr])
    o, ww = [], []
    for r in np.unique(rel[wr]):
        m = rel[wr] == r
        if y[m].sum() >= 20 and (1 - y[m]).sum() >= 20: o.append(roc_auc_score(y[m], gap[wr][m])); ww.append(m.sum())
    print(f"  AUROC (k>=1 'improvable' vs k=0 'unknown'): pooled {a:.3f} | inside relation {np.average(o, weights=ww):.3f} ({len(o)} relations)")
    mm = wr & (k >= 1); print(f"  among wrong rows of facts 1/5..4/5: Spearman(gap, k) = {spearmanr(gap[mm], k[mm])[0]:.3f}")
    # share of improvable (k>=1) among wrong rows, by gap quintile
    q = np.quantile(gap[wr], [0.2, 0.4, 0.6, 0.8]); b = np.digitize(gap[wr], q)
    print("  share of wrong rows whose fact is answered right under another wording (k>=1), by gap quintile (low -> high):", [round(float((k[wr][b == i] >= 1).mean()), 3) for i in range(5)])
