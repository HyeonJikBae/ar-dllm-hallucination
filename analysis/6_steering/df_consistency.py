"""How consistent are the per-fact differences d_f = mean(correct h) - mean(broken h) that make direction B?  (CPU)
Facts 1/5..4/5 with both correct and broken rows; layer-20 prompt-end step-0 states.
 - size of d_f, and its cosine with the mean direction of the OTHER facts (leave-one-out, so a fact is not compared with itself)
 - the same per k group, and the cosine between the group-mean directions of k = 1, 2, 3, 4
 - share of d_f's energy that lies along the common direction:  (d_f . B_loo)^2 / ||d_f||^2
 - null: the same numbers after shuffling the correct / broken labels inside each fact (500 shuffles is too slow; 20 shuffles)"""
import collections, json
import numpy as np
D = "data"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows])
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
byf = collections.defaultdict(lambda: collections.defaultdict(list))
for i in np.where((k >= 1) & (k <= 4) & np.isin(ty, ["correct", "broken"]))[0]: byf[fid[i]][ty[i]].append(int(i))
facts = [(f, v) for f, v in byf.items() if v["correct"] and v["broken"]]
X = {i: np.asarray(H[i, 0], dtype=np.float32) for _, v in facts for i in v["correct"] + v["broken"]}
kf = np.array([k[v["correct"][0]] for _, v in facts])
def diffs(shuffle_rng=None):
    out = []
    for _, v in facts:
        idx = v["correct"] + v["broken"]; nc = len(v["correct"])
        if shuffle_rng is not None: idx = list(shuffle_rng.permutation(idx))
        out.append(np.mean([X[i] for i in idx[:nc]], 0) - np.mean([X[i] for i in idx[nc:]], 0))
    return np.array(out)
def stats(d):
    S = d.sum(0); n = len(d); norms = np.linalg.norm(d, axis=1)
    loo = np.array([(S - d[j]) / (n - 1) for j in range(n)]); loo /= np.linalg.norm(loo, axis=1, keepdims=True)
    proj = (d * loo).sum(1); cos = proj / norms; share = proj ** 2 / norms ** 2
    return norms, cos, share, proj
d = diffs(); norms, cos, share, proj = stats(d)
B = d.mean(0)
print(f"facts {len(d)} | k: " + ", ".join(f"{g}/5: {int((kf == g).sum())}" for g in (1, 2, 3, 4)))
print(f"||mean d_f|| = {np.linalg.norm(B):.2f}   mean ||d_f|| = {norms.mean():.2f}   (ratio {np.linalg.norm(B) / norms.mean():.3f})")
print(f"all facts: cosine(d_f, leave-one-out mean) mean {cos.mean():.3f} median {np.median(cos):.3f} | share positive {(cos > 0).mean():.3f} | energy share along it {share.mean():.3f}")
print("per k group:")
for g in (4, 3, 2, 1):
    m = kf == g
    print(f"  {g}/5 (n={m.sum():>3d}): ||d_f|| mean {norms[m].mean():.2f} | cosine with leave-one-out mean: mean {cos[m].mean():.3f}, share positive {(cos[m] > 0).mean():.2f} | projection mean {proj[m].mean():.2f}")
gm = {g: d[kf == g].mean(0) for g in (1, 2, 3, 4)}; u = lambda v: v / np.linalg.norm(v)
print("cosine between group-mean directions:", {f"{a}/5-{b}/5": round(float(u(gm[a]) @ u(gm[b])), 2) for a in (1, 2, 3, 4) for b in (1, 2, 3, 4) if a < b})
print("group-mean length:", {f"{g}/5": round(float(np.linalg.norm(gm[g])), 2) for g in (1, 2, 3, 4)})
rng = np.random.default_rng(0); nc_, ns_, np_ = [], [], []
for _ in range(20):
    dn = diffs(rng); n2, c2, s2, _ = stats(dn); nc_.append(c2.mean()); ns_.append(np.linalg.norm(dn.mean(0)) / n2.mean()); np_.append((c2 > 0).mean())
print(f"NULL (labels shuffled inside each fact, 20x): cosine mean {np.mean(nc_):.3f} (max {np.max(nc_):.3f}) | share positive {np.mean(np_):.3f} | ratio ||mean|| / mean||d_f|| {np.mean(ns_):.3f}")
