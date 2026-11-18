"""Does the layer-20 SAE keep the signals we care about?  Dream-v0-Base-7B, SAE = sae/dream_pilot_200M (16k, trained on masked SlimPajama).  CPU.
Every hidden state h is split exactly into  h = r + e,  r = SAE reconstruction (a @ W_dec + b_dec), e = error (what the SAE misses).
 1) reconstruction quality at our positions (step 0 and step 8; prompt end / end of question / end of subject):
    FVU = sum||h - r||^2 / sum||h - mean h||^2 (0 = perfect), mean cosine(h, r), L0 (latents on per row), latents ever on.
 2) where the steering direction lives: per fact (1/5..4/5) d_f = mean(correct) - mean(wrong), averaged over facts -> B_h, and the same on r and on e
    (B_h = B_r + B_e exactly).  Share of ||B_h||^2 carried by r and by e, cosines.  Same for the "know" direction (5/5 minus 0/5, inside relations).
 3) where the readout lives: 5-fold subject cross-fitted logistic probes on h, on r, on e and on the SAE activations a
      mid  : correct vs wrong, trained on facts 1/5..4/5, scored by within-fact AUROC
      know : 5/5 rows vs 0/5 rows, pooled AUROC
Correct vs wrong only (wrong types merged).  Usage (repo root): python3 analysis/7_sae_vs_hidden/sae_diagnosis.py"""
import collections, json, os, sys
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "analysis"); import sae_io
torch.set_num_threads(16)
D = "data"; NF = 5
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); order = {x: j for j, x in enumerate(subs)}; grp = np.array([order[x] % NF for x in sub])
k = np.array([r["k_correct"] for r in rows]); cor = np.array([r["correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); rel = np.array([r["relation"] for r in rows])
mid = (k >= 1) & (k <= 4); ext = (k == 0) | (k == 5)
enc = sae_io.dream_sae(); Wd = enc.W_dec; bd = sae_io._load(sae_io.DREAM)["b_dec"]
F = f"{D}/1_raw/dream_hidden_steps"

def split(X):
    A, R = [], []
    for s in range(0, len(X), 4096):
        x = torch.from_numpy(X[s:s + 4096]); a = enc(x); A.append(a.to_sparse()); R.append((a @ Wd + bd).numpy())
    return A, np.concatenate(R)

def mean_diff(Z):
    byf = collections.defaultdict(lambda: [[], []])
    for i in np.where(mid)[0]: byf[fid[i]][int(cor[i])].append(i)
    return np.mean([Z[c].mean(0) - Z[w].mean(0) for w, c in byf.values() if w and c], 0)

def know_dir(Z):
    ds, wt = [], []
    for r_ in np.unique(rel):
        a, b = (rel == r_) & (k == 5), (rel == r_) & (k == 0)
        if a.sum() >= 20 and b.sum() >= 20: ds.append(Z[a].mean(0) - Z[b].mean(0)); wt.append(min(a.sum(), b.sum()))
    return np.average(ds, axis=0, weights=wt)

def fit(Xtr, ytr, cap=20000):
    if len(Xtr) > cap: s = np.random.default_rng(0).choice(len(Xtr), cap, replace=False); Xtr, ytr = Xtr[s], ytr[s]
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=0.001, max_iter=500).fit(sc.transform(Xtr), ytr); return lambda Z: m.decision_function(sc.transform(Z))

def within(score, m):
    per = []
    for u in np.unique(fid[m]):
        mm = m & (fid == u)
        if cor[mm].min() != cor[mm].max(): per.append(roc_auc_score(cor[mm], score[mm]))
    return np.mean(per)

def probes(Z):
    s_mid = np.full(len(Z), np.nan); s_know = np.full(len(Z), np.nan)
    for g in range(NF):
        tr, te = grp != g, grp == g
        s_mid[te & mid] = fit(Z[tr & mid], cor[tr & mid])(Z[te & mid])
        s_know[te & ext] = fit(Z[tr & ext], k[tr & ext] == 5)(Z[te & ext])
    return within(s_mid, mid), roc_auc_score(k[ext] == 5, s_know[ext])

print("1) reconstruction quality (all 55,225 rows)")
print("   position      step  FVU     cos(h,r)  L0     latents ever on")
keep = {}
for pos in ("prompt_last", "q_last", "subj_last"):
    H = np.load(f"{F}/hid_{pos}.npy", mmap_mode="r")
    for si, step in ((0, 0), (1, 8)):
        X = np.asarray(H[:, si], dtype=np.float32)
        if pos == "subj_last": X = X[np.abs(X).sum(1) > 0]          # subject not found -> zeros
        A, R = split(X)
        L0 = np.concatenate([np.bincount(a.indices()[0].numpy(), minlength=a.shape[0]) for a in A])
        ever = torch.zeros(Wd.shape[0], dtype=torch.bool)
        for a in A: ever[a.indices()[1].unique()] = True
        fvu = ((X - R) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
        cs = ((X * R).sum(1) / np.linalg.norm(X, axis=1) / np.linalg.norm(R, axis=1)).mean()
        print(f"   {pos:13s} {step:4d}  {fvu:.3f}   {cs:.3f}     {L0.mean():5.1f}  {int(ever.sum())}", flush=True)
        if pos == "prompt_last" and step == 0: keep = dict(X=X, R=R, A=A)

X, R = keep["X"], keep["R"]; E = X - R
print("\n2) where the directions live (prompt end, step 0).  B = within-fact correct - wrong, facts 1/5..4/5;  K = 5/5 - 0/5 inside relations")
for name, f in (("B", mean_diff), ("K", know_dir)):
    vh, vr, ve = f(X), f(R), f(E); n2 = (vh ** 2).sum()
    cos = lambda u, v: u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    print(f"   {name}: ||{name}_h|| {np.linalg.norm(vh):.2f} | share of ||{name}_h||^2 along it: from r {vh @ vr / n2:.2f}, from e {vh @ ve / n2:.2f} | "
          f"||{name}_r|| {np.linalg.norm(vr):.2f} ||{name}_e|| {np.linalg.norm(ve):.2f} | cos(h,r) {cos(vh, vr):.2f} cos(h,e) {cos(vh, ve):.2f}")

print("\n3) probes, 5-fold by subject (prompt end, step 0).   mid = within-fact AUROC correct vs wrong (1/5..4/5);   know = AUROC 5/5 vs 0/5")
Adense = torch.cat([a.to_dense() for a in keep["A"]]).numpy(); alive = Adense.any(0); Adense = Adense[:, alive]
for name, Z in (("h (raw hidden)", X), ("r (SAE reconstruction)", R), ("e (SAE error)", E), (f"a (SAE activations, {alive.sum()} latents)", Adense)):
    m_, k_ = probes(Z)
    print(f"   {name:36s} mid {m_:.3f} | know {k_:.3f}", flush=True)
