"""Which layer AND which denoising step carry the signals best?  Prompt-end state (extract_layers_steps.py), steps 0 / 4 / 8 / 16 / 31.  CPU.
Same probes and folds as layer_probes.py:
  mid  : correct vs wrong, trained on facts 1/5..4/5, within-fact AUROC
  know : 5/5 vs 0/5 (5,000 sampled 0/5 rows), mean of per-relation AUROCs
Baseline per step (no hidden state): logistic regression on the answer-slot confidence (top-1 log-prob and margin of all 32 slots) of every step up to s.
After step 0 part of the answer is already written (correct answers fill their first slot early), so later steps partly read the answer itself.
Only rows whose re-generated canvas is identical to the stored run are used.
Usage (repo root): python3 analysis/7_sae_vs_hidden/layer_step_probes.py [layers, e.g. 0,8,16,20,24,27]"""
import collections, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = "data"; NF = 5; STEPS = [0, 4, 8, 16, 31]
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); order = {x: j for j, x in enumerate(subs)}; grp_all = np.array([order[x] % NF for x in sub])
keep = np.load(f"{D}/1_raw/dream_hidden_layers/rows_layers_steps.npy"); same = np.load(f"{D}/1_raw/dream_hidden_layers/same_canvas_layers_steps.npy")
H = np.load(f"{D}/1_raw/dream_hidden_layers/hid_prompt_last_layers_steps.npy", mmap_mode="r")
use = np.where(same)[0]; ri = keep[use]
k = np.array([rows[i]["k_correct"] for i in ri]); cor = np.array([rows[i]["correct"] for i in ri]); fid = np.array([rows[i]["sample_id"] for i in ri]); rel = np.array([rows[i]["relation"] for i in ri]); grp = grp_all[ri]
mid = (k >= 1) & (k <= 4); ext = (k == 0) | (k == 5)
layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 4, 8, 12, 14, 16, 18, 20, 22, 24, 27]
print(f"rows used {len(use)} of {len(keep)} saved (canvas identical to the stored run) | 1/5..4/5 rows {mid.sum()}, 5/5 {(k == 5).sum()}, 0/5 {(k == 0).sum()}", flush=True)

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
def run(X, know=True):
    s_mid = np.full(len(X), np.nan); s_kn = np.full(len(X), np.nan)
    for g in range(NF):
        tr, te = grp != g, grp == g
        s_mid[te & mid] = fit(X[tr & mid], cor[tr & mid])(X[te & mid])
        if know: s_kn[te & ext] = fit(X[tr & ext], k[tr & ext] == 5)(X[te & ext])
    return within(s_mid), (rel_auc(s_kn) if know else np.nan)

conf = [np.load(f"{D}/1_raw/dream_hidden_steps/{n}.npy", mmap_mode="r") for n in ("conf_lp", "conf_mg")]
print("\nbaseline: answer-slot confidence of steps 0..s (no hidden state)")
for s in STEPS:
    Xc = np.concatenate([np.asarray(c[ri, : s + 1], dtype=np.float32).reshape(len(ri), -1) for c in conf], 1)
    m_, k_ = run(Xc); print(f"  step {s:2d}: mid {m_:.3f} | know {k_:.3f}", flush=True)
res = {}
for si, s in enumerate(STEPS):
    print(f"\nstep {s}:  layer | mid (within-fact) | know (in-relation)", flush=True)
    for L in layers:
        X = np.asarray(H[use, si, L], dtype=np.float32); res[(s, L)] = run(X)
        print(f"          {L:5d} | {res[(s, L)][0]:.3f}             | {res[(s, L)][1]:.3f}", flush=True)
print("\nmid (within-fact AUROC): rows = layer, columns = step " + " ".join(f"{s:>6d}" for s in STEPS))
for L in layers: print(f"  layer {L:2d}                                    " + " ".join(f"{res[(s, L)][0]:6.3f}" for s in STEPS))
print("know (in-relation AUROC): rows = layer, columns = step " + " ".join(f"{s:>6d}" for s in STEPS))
for L in layers: print(f"  layer {L:2d}                                    " + " ".join(f"{res[(s, L)][1]:6.3f}" for s in STEPS))
