"""Pairwise cosine between INDIVIDUAL per-fact differences d_f, within and across k groups (1/5..4/5), and the same after shuffling correct/broken labels inside each fact (null).  CPU."""
import collections, json
import numpy as np
D = "data"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows])
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
byf = collections.defaultdict(lambda: collections.defaultdict(list))
for i in np.where((k >= 1) & (k <= 4) & np.isin(ty, ["correct", "broken"]))[0]: byf[fid[i]][ty[i]].append(int(i))
facts = [v for v in byf.values() if v["correct"] and v["broken"]]
X = {i: np.asarray(H[i, 0], dtype=np.float32) for v in facts for i in v["correct"] + v["broken"]}
kf = np.array([k[v["correct"][0]] for v in facts])
def diffs(rng=None):
    out = []
    for v in facts:
        idx = v["correct"] + v["broken"]; nc = len(v["correct"])
        if rng is not None: idx = list(rng.permutation(idx))
        out.append(np.mean([X[i] for i in idx[:nc]], 0) - np.mean([X[i] for i in idx[nc:]], 0))
    d = np.array(out); return d / np.linalg.norm(d, axis=1, keepdims=True)
def table(u):
    C = u @ u.T; T = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            ma, mb = kf == a + 1, kf == b + 1; S = C[np.ix_(ma, mb)]
            T[a, b] = S[~np.eye(len(S), dtype=bool)].mean() if a == b else S.mean()
    return T, C
T, C = table(diffs())
rng = np.random.default_rng(0); Tn = np.mean([table(diffs(rng))[0] for _ in range(20)], 0)
print("mean pairwise cosine between individual d_f (rows/cols = k):")
print("        " + "  ".join(f"{g}/5   " for g in (1, 2, 3, 4)))
for a in range(4): print(f"  {a + 1}/5  " + "  ".join(f"{T[a, b]:.3f}" for b in range(4)) + "     | null: " + "  ".join(f"{Tn[a, b]:.3f}" for b in range(4)))
off = C[~np.eye(len(C), dtype=bool)]
print(f"all pairs: mean {off.mean():.3f}, share of pairs with positive cosine {(off > 0).mean():.3f}, 90th percentile {np.percentile(off, 90):.3f}, max {off.max():.3f}")
