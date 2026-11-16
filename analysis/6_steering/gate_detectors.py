"""Can we tell BEFORE generation whether a row will be answered correctly?  Cross-fitted by the same 5 subject folds as direction B (train on 4 folds, score the 5th).  CPU.
Input: step-0 prompt-end layer-20 state h.   Detectors:
  P_all  : logistic probe, correct vs wrong, trained on rows of ALL facts (0/5..5/5)          -> "will this row be right?"
  P_mid  : logistic probe, correct vs wrong, trained only on rows of facts 1/5..4/5          -> "known fact, but right or wrong this time?"
  P_know : logistic probe, fact known (k >= 1) vs not (k = 0), trained on all rows
  axis   : h . B (the steering direction itself; higher = more on the correct side)
Reports AUROC on held-out rows; saves every row's scores to analysis/6_steering/gate_scores.npz for the gating simulation."""
import collections, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = "data"; NF = 5
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); order = {x: j for j, x in enumerate(subs)}; grp = np.array([order[x] % NF for x in sub])
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
cor = ty == "correct"
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r"); X = np.asarray(H[:, 0], dtype=np.float32)
Bd = np.load("analysis/6_steering/B_directions.npy")
N = len(rows); S = {n: np.full(N, np.nan) for n in ("P_all", "P_mid", "P_know", "axis")}
def fit(Xtr, ytr, cap=20000, seed=0):
    if len(Xtr) > cap: s = np.random.default_rng(seed).choice(len(Xtr), cap, replace=False); Xtr, ytr = Xtr[s], ytr[s]
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=0.001, max_iter=500).fit(sc.transform(Xtr), ytr); return lambda Z: m.decision_function(sc.transform(Z))
for g in range(NF):
    tr, te = grp != g, grp == g
    S["P_all"][te] = fit(X[tr], cor[tr])(X[te])
    trm = tr & mid; S["P_mid"][te] = fit(X[trm], cor[trm])(X[te])
    S["P_know"][te] = fit(X[tr], k[tr] >= 1)(X[te])
    S["axis"][te] = X[te] @ Bd[g]
    print("fold", g, "done", flush=True)
np.savez("analysis/6_steering/gate_scores.npz", **S)
def within(score, m, pos):
    per = []
    for u in np.unique(fid[m]):
        mm = m & (fid == u)
        if pos[mm].min() != pos[mm].max(): per.append(roc_auc_score(pos[mm], score[mm]))
    return np.mean(per), len(per)
print("\nAUROC (held-out rows).  'right vs wrong' = correct vs any wrong answer")
for n in ("P_all", "P_mid", "axis", "P_know"):
    s = S[n]; a_all = roc_auc_score(cor, s); a_mid = roc_auc_score(cor[mid], s[mid]); w, nf = within(s, mid, cor)
    cb = mid & np.isin(ty, ["correct", "broken"]); ce = mid & np.isin(ty, ["correct", "entity"])
    wb, _ = within(s, cb, cor); we, _ = within(s, ce, cor)
    print(f"  {n:<7s} all rows {a_all:.3f} | facts 1-4/5 pooled {a_mid:.3f} | 1-4/5 within-fact {w:.3f} ({nf} facts) | within-fact correct-vs-broken {wb:.3f}, correct-vs-entity {we:.3f}")
print(f"  P_know: known(k>=1) vs unknown(k=0) AUROC {roc_auc_score(k >= 1, S['P_know']):.3f} | 5/5 vs 1-4/5 rows {roc_auc_score(k[k >= 1] == 5, S['P_know'][k >= 1]):.3f}")
