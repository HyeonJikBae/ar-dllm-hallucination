"""Gating, looked at directly: inside facts 1/5..4/5, steer the X% of rows the detector ranks as most likely wrong, and compare with random choice and the oracle.  Plus per-group effects (0/5, 5/5 samples)."""
import collections, json
import numpy as np
rows = [json.loads(l) for l in open("data/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
k_all = np.array([r["k_correct"] for r in rows])
R = collections.defaultdict(dict)
for l in open("analysis/6_steering/results_caa_all.jsonl", encoding="utf-8"): r = json.loads(l); R[r["cond"]][r["row"]] = r["new_type"]
idx = np.array(sorted(R["base"])); k = k_all[idx]
T = {c: np.array([R[c][i] for i in idx]) for c in R}; ok = {c: (T[c] == "correct").astype(float) for c in T}
print("per group: correct share  base / B c0.5 / B c1   (and broken share)")
for name, m in (("0/5 sample", k == 0), ("1/5", k == 1), ("2/5", k == 2), ("3/5", k == 3), ("4/5", k == 4), ("5/5 sample", k == 5)):
    print(f"  {name:<11s} n={m.sum():>4d}  correct {ok['base'][m].mean():.3f} / {ok['B c0.5'][m].mean():.3f} / {ok['B c1'][m].mean():.3f}   broken {np.mean(T['base'][m] == 'broken'):.3f} / {np.mean(T['B c0.5'][m] == 'broken'):.3f} / {np.mean(T['B c1'][m] == 'broken'):.3f}")
S = np.load("analysis/6_steering/gate_scores.npz"); mid = (k >= 1) & (k <= 4)
print("\ninside facts 1/5..4/5 (n=%d): accuracy when steering the X%% of rows ranked most likely wrong   [base %.1f%%]" % (mid.sum(), 100 * ok["base"][mid].mean()))
rng = np.random.default_rng(0)
for c in ("B c0.5", "B c1"):
    b, s_ = ok["base"][mid], ok[c][mid]; gain = s_ - b
    print(f"  {c}:")
    for X in (0.2, 0.4, 0.6, 0.8, 1.0):
        n = int(round(X * len(b))); line = []
        for name in ("P_mid", "axis", "P_all"):
            sc = S[name][idx][mid]; sel = np.argsort(sc)[:n]; line.append(f"{name} {100 * (b.mean() + gain[sel].sum() / len(b)):.1f}")
        rand = np.mean([b.mean() + gain[rng.choice(len(b), n, replace=False)].sum() / len(b) for _ in range(200)])
        orc = np.argsort(-gain)[:n]
        print(f"    steer {int(X * 100):>3d}%: random {100 * rand:.1f} | " + " | ".join(line) + f" | oracle {100 * (b.mean() + gain[orc].sum() / len(b)):.1f}")
