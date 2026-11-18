"""Which layer carries the signals best?  Prompt-end state at denoising step 0, every layer (extract_layers.py).  CPU.
Per layer, 5-fold subject cross-fitting (same folds as analysis/6_steering):
  mid    : logistic probe correct vs wrong trained on facts 1/5..4/5, within-fact AUROC        (will THIS wording be right?)
  axis   : h . B_L, B_L = within-fact mean(correct) - mean(wrong) from the other folds, within-fact AUROC   (the steering direction as a readout)
  know   : logistic probe 5/5 rows vs 0/5 rows, pooled AUROC, and the mean of per-relation AUROCs   (does the model know the fact?)
  |B_L| / mean||h||  : how large the steering direction is relative to the state at that layer
Correct vs wrong only.   Usage (repo root): python3 analysis/7_sae_vs_hidden/layer_probes.py [layers, e.g. 0,4,8]"""
import collections, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = "data"; NF = 5
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); order = {x: j for j, x in enumerate(subs)}; grp = np.array([order[x] % NF for x in sub])
k = np.array([r["k_correct"] for r in rows]); cor = np.array([r["correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); rel = np.array([r["relation"] for r in rows])
mid = (k >= 1) & (k <= 4); ext = (k == 0) | (k == 5)
H = np.load(f"{D}/1_raw/dream_hidden_layers/hid_prompt_last_layers.npy", mmap_mode="r")
layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(H.shape[1]))
byf = collections.defaultdict(lambda: [[], []])
for i in np.where(mid)[0]: byf[fid[i]][int(cor[i])].append(i)
pairs = [(w, c, grp[c[0]]) for w, c in byf.values() if w and c]

def fit(Xtr, ytr, cap=20000):
    if len(Xtr) > cap: s = np.random.default_rng(0).choice(len(Xtr), cap, replace=False); Xtr, ytr = Xtr[s], ytr[s]
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=0.001, max_iter=500).fit(sc.transform(Xtr), ytr); return lambda Z: m.decision_function(sc.transform(Z))
def within(score):
    per = []
    for u in np.unique(fid[mid]):
        mm = mid & (fid == u)
        if cor[mm].min() != cor[mm].max(): per.append(roc_auc_score(cor[mm], score[mm]))
    return np.mean(per)
def rel_auc(score):
    out = []
    for r_ in np.unique(rel[ext]):
        m = ext & (rel == r_)
        if (k[m] == 5).sum() >= 20 and (k[m] == 0).sum() >= 20: out.append(roc_auc_score(k[m] == 5, score[m]))
    return np.mean(out)

print("layer | mid (probe) | axis h.B | know pooled | know in-relation | |B|/mean||h|| | mean||h||", flush=True)
for L in layers:
    X = np.asarray(H[:, L], dtype=np.float32); nh = np.linalg.norm(X, axis=1).mean()
    s_mid = np.full(len(X), np.nan); s_ax = np.full(len(X), np.nan); s_kn = np.full(len(X), np.nan); bl = []
    for g in range(NF):
        tr, te = grp != g, grp == g
        s_mid[te & mid] = fit(X[tr & mid], cor[tr & mid])(X[te & mid])
        B = np.mean([X[c].mean(0) - X[w].mean(0) for w, c, gg in pairs if gg != g], 0); bl.append(np.linalg.norm(B))
        s_ax[te & mid] = X[te & mid] @ (B / np.linalg.norm(B))
        s_kn[te & ext] = fit(X[tr & ext], k[tr & ext] == 5)(X[te & ext])
    print(f"{L:5d} | {within(s_mid):.3f}       | {within(s_ax):.3f}    | {roc_auc_score(k[ext] == 5, s_kn[ext]):.3f}       | {rel_auc(s_kn):.3f}            | {np.mean(bl) / nh:.3f}        | {nh:.1f}", flush=True)
