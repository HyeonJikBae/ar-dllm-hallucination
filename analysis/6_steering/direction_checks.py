"""Where does the steering direction come from, and would another mean difference do the same?  CPU only.  Dream, layer 20, prompt-end token, step-0 states.
Correct vs wrong only (wrong = entity + broken + abstain).
 1) per k group (facts 1/5..4/5): direction_k = unit( mean over facts of k of  d_f = mean(correct h) - mean(wrong h) ).  Cosine between the four directions,
    split-half cosine inside each k (stability of that direction, 20 random halvings of the facts), and the same after shuffling correct / wrong inside each fact (null).
 2) the all-k direction with and without the 1/5 facts, and both against direction B (B_directions.npy from caa_all_rows.py, one per subject fold).
 3) "know" direction K = mean(h of 5/5 rows) - mean(h of 0/5 rows), and K_rel = the same inside each relation, averaged (relations with >= 20 rows on both sides):
    cosine with B, plus the relation mix of 5/5 and 0/5 rows (why K and B differ).
Usage (repo root): python3 analysis/6_steering/direction_checks.py"""
import collections, json
import numpy as np

rows = [json.loads(l) for l in open("data/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
k = np.array([r["k_correct"] for r in rows]); cor = np.array([r["correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); rel = np.array([r["relation"] for r in rows])
X = np.asarray(np.load("data/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")[:, 0], dtype=np.float32)
B = np.load("analysis/6_steering/B_directions.npy")
unit = lambda v: v / np.linalg.norm(v); rng = np.random.default_rng(0)

byf = collections.defaultdict(lambda: [[], []])                      # fact -> [wrong rows, correct rows]
for i in np.where((k >= 1) & (k <= 4))[0]: byf[fid[i]][int(cor[i])].append(i)
facts = [(k[c[0]], w, c) for w, c in byf.values() if w and c]
D = {g: np.array([X[c].mean(0) - X[w].mean(0) for kk, w, c in facts if kk == g]) for g in range(1, 5)}
M = {g: unit(D[g].mean(0)) for g in D}

print("1) per-k directions (correct - wrong, within fact)   facts per k:", {g: len(D[g]) for g in D})
print("   cosine        " + "  ".join(f"{g}/5  " for g in range(1, 5)))
for a in range(1, 5): print(f"   {a}/5        " + "  ".join(f"{M[a] @ M[b]:.2f}" for b in range(1, 5)))
print("   split-half cosine inside k (20 halvings):", end="")
for g in range(1, 5):
    s = []
    for _ in range(20):
        p = rng.permutation(len(D[g])); h = len(D[g]) // 2; s.append(unit(D[g][p[:h]].mean(0)) @ unit(D[g][p[h:]].mean(0)))
    print(f"  {g}/5 {np.mean(s):.2f}", end="")
print()
Mn = {}
for g in range(1, 5):
    acc = []
    for kk, w, c in facts:
        if kk != g: continue
        p = rng.permutation(np.array(w + c)); acc.append(X[p[:len(c)]].mean(0) - X[p[len(c):]].mean(0))
    Mn[g] = unit(np.mean(acc, 0))
print("   null (labels shuffled inside each fact), cosine between k groups:", " ".join(f"{a}-{b} {Mn[a] @ Mn[b]:.2f}" for a in range(1, 5) for b in range(a + 1, 5)))

ALL = unit(np.concatenate([D[g] for g in D]).mean(0)); NO1 = unit(np.concatenate([D[g] for g in (2, 3, 4)]).mean(0))
print("\n2) all-k direction vs each k:", " ".join(f"{g}/5 {ALL @ M[g]:.2f}" for g in D))
print(f"   all-k vs without 1/5: {ALL @ NO1:.3f}")
print("   B (5 folds) vs all-k:      ", " ".join(f"{ALL @ unit(b):.2f}" for b in B))
print("   B (5 folds) vs without 1/5:", " ".join(f"{NO1 @ unit(b):.2f}" for b in B))

raw = X[k == 5].mean(0) - X[k == 0].mean(0); K = unit(raw)
ds, wt = [], []
for r_ in np.unique(rel):
    a, b = (rel == r_) & (k == 5), (rel == r_) & (k == 0)
    if a.sum() >= 20 and b.sum() >= 20: ds.append(X[a].mean(0) - X[b].mean(0)); wt.append(min(a.sum(), b.sum()))
Kr = unit(np.average(ds, axis=0, weights=wt))
print(f"\n3) K = mean(5/5) - mean(0/5): raw length {np.linalg.norm(raw):.1f}")
print("   B (5 folds) vs K:    ", " ".join(f"{K @ unit(b):.2f}" for b in B))
print(f"   B (5 folds) vs K_rel ({len(ds)} relations):", " ".join(f"{Kr @ unit(b):.2f}" for b in B), f"| K vs K_rel {K @ Kr:.2f}")
c5, c0 = collections.Counter(rel[k == 5]), collections.Counter(rel[k == 0]); n5, n0 = sum(c5.values()), sum(c0.values())
print("   relation mix (share of rows):   relation          5/5     0/5")
for r_ in sorted(set(c5) | set(c0), key=lambda x: -c5[x] / n5): print(f"   {'':30s}{r_:16s} {c5[r_] / n5 * 100:5.1f}%  {c0[r_] / n0 * 100:5.1f}%")
