"""Is "knowing" a continuous quantity?  (Dream, prompt end, layer 20, step 0)  No binary classes: regress k = number of correct paraphrases (0..5).

Features : raw hidden state at the last prompt token.   Model: ridge regression, fitted on dev subjects, scored on test subjects.
Outputs  : Spearman(prediction, k) overall / inside each relation / inside 1/5..4/5 facts only; mean prediction per k;
           share of correct / entity / broken rows along deciles of the prediction (relation-adjusted);
           inside a fact: does the continuous score also rank correct rows above wrong rows?
Usage (repo root): python3 analysis/3_continuous_knowledge/continuous.py
"""
import json, warnings, collections
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
k = np.array([r["k_correct"] for r in rows]); rel = np.array([r["relation"] for r in rows]); cor = np.array([r["correct"] for r in rows])
ty = np.array([r["type3"] for r in rows]); fact = np.array([r["sample_id"] for r in rows]); sub = np.array([popqa[f] for f in fact])
X = np.asarray(np.load(f"{F}/hid_prompt_last.npy", mmap_mode="r")[:, 0], dtype=np.float32)
lp = np.asarray(np.load(f"{F}/conf_lp.npy", mmap_mode="r")[:, 0, 0], dtype=np.float32)
rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
mu, sd = X[dev].mean(0), X[dev].std(0) + 1e-6; Xs = (X - mu) / sd
print("rows", len(rows), "| dev", dev.sum(), "test", te.sum())
best = None
for a in (3e3, 3e4, 3e5):          # picked on a dev-internal split, never on test
    d2 = dev & np.isin(sub, subs[: len(subs) // 4]); d1 = dev & ~d2
    s = spearmanr(Ridge(alpha=a).fit(Xs[d1], k[d1]).predict(Xs[d2]), k[d2])[0]
    print(f"  alpha {a:>8.0f}  dev-internal Spearman {s:.3f}")
    if best is None or s > best[0]: best = (s, a)
a = best[1]; pred = Ridge(alpha=a).fit(Xs[dev], k[dev]).predict(Xs); print("alpha picked on dev:", a)
m = te
print(f"\nSpearman(prediction, k) on test subjects: all rows {spearmanr(pred[m], k[m])[0]:.3f} | first-token log-prob alone {spearmanr(lp[m], k[m])[0]:.3f}")
sr, w = [], []
for r in np.unique(rel):
    mm = m & (rel == r)
    if len(np.unique(k[mm])) > 2 and mm.sum() >= 200: sr.append(spearmanr(pred[mm], k[mm])[0]); w.append(mm.sum())
print(f"inside each relation (weighted mean of {len(sr)}): {np.average(sr, weights=w):.3f}")
mid = m & (k >= 1) & (k <= 4)
print(f"only 1/5..4/5 facts (the 'partly known' range): {spearmanr(pred[mid], k[mid])[0]:.3f} (n={mid.sum()})")
print("\nmean prediction per k (test):", {int(g): round(float(pred[m & (k == g)].mean()), 2) for g in range(6)})
print("correlation of the continuous score with the first-token log-prob:", round(float(np.corrcoef(pred[m], lp[m])[0, 1]), 2))
# deciles of the continuous score: what happens to correct / entity / broken
q = np.quantile(pred[m], np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, pred, side="right") - 1, 0, 9)
print(f"\n{'decile':>6s} {'mean k':>7s} {'correct':>8s} {'entity':>7s} {'broken':>7s} {'abstain':>8s} | broken share of wrong")
for i in range(10):
    mm = m & (b == i); n = mm.sum(); wr = mm & ~cor
    print(f"{i + 1:6d} {k[mm].mean():7.2f} {100 * cor[mm].mean():7.1f}% {100 * (ty[mm] == 'entity').mean():6.1f}% {100 * (ty[mm] == 'broken').mean():6.1f}% {100 * (ty[mm] == 'abstain').mean():7.1f}% | {100 * (ty[wr] == 'broken').mean():5.1f}%")
# inside a fact: continuous score vs correct / wrong rows
aucs = []
for f in np.unique(fact[mid]):
    mm = mid & (fact == f); y = cor[mm]
    if 0 < y.sum() < len(y):
        p, n_ = pred[mm][y], pred[mm][~y]; aucs.append((p[:, None] > n_[None, :]).mean() + 0.5 * (p[:, None] == n_[None, :]).mean())
print(f"\ninside a 1/5..4/5 fact, does the knowledge score rank correct rows above wrong rows? AUROC {np.mean(aucs):.3f} ({len(aucs)} facts)")
json.dump({"alpha": a, "spearman_all": float(spearmanr(pred[m], k[m])[0]), "spearman_mid": float(spearmanr(pred[mid], k[mid])[0])}, open("analysis/3_continuous_knowledge/results_dream.json", "w"))
