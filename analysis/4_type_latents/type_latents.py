"""Can SAE latents that are simply ON / OFF (activation > 0, strength ignored) tell the three wrong-answer types apart?  (Dream, prompt end, layer 20, step 0)
Wrong rows only: entity / broken / abstain.  Subjects split dev / test (50/50).
For every pair of types (A, B): firing fraction of each latent on dev rows of A and of B; separation = f_A - f_B (the paper's separation score);
the top-K latents for A and for B give a binary score  (#A-latents on) - (#B-latents on);  AUROC of that score on test rows, pooled and inside each relation.
Usage (repo root): python3 analysis/4_type_latents/type_latents.py
"""
import json, sys, os, warnings
import numpy as np, torch
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); torch.set_num_threads(16)
sys.path.insert(0, "analysis")
from sae_io import dream_sae
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
keep = np.array([i for i, r in enumerate(rows) if not r["correct"]])
ty = np.array([rows[i]["type3"] for i in keep]); rel = np.array([rows[i]["relation"] for i in keep]); sub = np.array([popqa[rows[i]["sample_id"]] for i in keep])
rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
enc = dream_sae(); H = np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")
on = np.zeros((len(keep), 16384), bool)
with torch.no_grad():
    for b in range(0, len(keep), 2048):
        on[b:b + 2048] = (enc(torch.from_numpy(np.asarray(H[keep[b:b + 2048], 0], dtype=np.float32))) > 0).numpy()
print("wrong rows", len(keep), {t: int((ty == t).sum()) for t in ("entity", "broken", "abstain")}, "| mean latents on", round(float(on.sum(1).mean()), 1))
def within(score, y, m):
    out, w = [], []
    for r in np.unique(rel[m]):
        k = m & (rel == r)
        if y[k].sum() >= 20 and (1 - y[k]).sum() >= 20: out.append(roc_auc_score(y[k], score[k])); w.append(k.sum())
    return (np.average(out, weights=w) if out else float("nan")), len(out)
for A, B in (("broken", "entity"), ("abstain", "entity"), ("abstain", "broken")):
    da, db = dev & (ty == A), dev & (ty == B); ta, tb = te & (ty == A), te & (ty == B)
    fa, fb = on[da].mean(0), on[db].mean(0); sep = fa - fb
    print(f"\n=== {A} (dev {da.sum()}, test {ta.sum()}) vs {B} (dev {db.sum()}, test {tb.sum()})")
    ia, ib = np.argsort(-sep), np.argsort(sep)
    print("  top latents for", A, ":", [(int(j), round(float(fa[j]), 2), round(float(fb[j]), 2)) for j in ia[:4]], "(id, f_A, f_B)")
    print("  top latents for", B, ":", [(int(j), round(float(fa[j]), 2), round(float(fb[j]), 2)) for j in ib[:4]])
    print(f"  best separation: for {A} {sep[ia[0]]:.2f} | for {B} {-sep[ib[0]]:.2f}")
    m = te & np.isin(ty, [A, B]); y = (ty == A).astype(int)
    for K in (1, 5, 20, 100):
        s = on[:, ia[:K]].sum(1) - on[:, ib[:K]].sum(1)
        a = roc_auc_score(y[m], s[m]); w, nr = within(s, y, m)
        print(f"  top-{K:<3d} on/off score: AUROC pooled {a:.3f} | inside each relation {w:.3f} ({nr} relations)")
