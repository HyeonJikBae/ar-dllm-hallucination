"""Steer only when a detector says the row will probably be wrong.  Offline simulation from results_caa_all.jsonl (base / B c0.5 / B c1 for every row of facts 1/5..4/5,
plus 1,000 rows each of 0/5 and 5/5 facts) and gate_scores.npz (cross-fitted detectors).
Whole-data accuracy weights the 0/5 and 5/5 samples up to their real sizes (44,110 and 5,165 rows).
Gate thresholds (and c) are chosen on the OTHER 4 folds and applied to each fold (nested), so the reported numbers are not tuned on the rows they are scored on."""
import collections, json
import numpy as np
D = "data"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
k_all = np.array([r["k_correct"] for r in rows])
R = collections.defaultdict(dict); fold = {}
for l in open("analysis/6_steering/results_caa_all.jsonl", encoding="utf-8"):
    r = json.loads(l); R[r["cond"]][r["row"]] = r["new_type"] == "correct"; fold[r["row"]] = r["fold"]
idx = np.array(sorted(R["base"])); assert all(len(R[c]) == len(idx) for c in ("B c0.5", "B c1"))
k = k_all[idx]; f = np.array([fold[i] for i in idx])
w = np.where(k == 0, 44110 / (k == 0).sum(), np.where(k == 5, 5165 / (k == 5).sum(), 1.0))
base = np.array([R["base"][i] for i in idx], float); st = {c: np.array([R[c][i] for i in idx], float) for c in ("B c0.5", "B c1")}
S = np.load("analysis/6_steering/gate_scores.npz"); sc = {n: S[n][idx] for n in S.files}
mid = (k >= 1) & (k <= 4)
def acc(steer_mask, c, m=None):
    y = np.where(steer_mask, st[c], base); m = np.ones(len(y), bool) if m is None else m
    return (y[m] * w[m]).sum() / w[m].sum()
def report(name, steer_mask, c):
    y = np.where(steer_mask, st[c], base)
    fixed = ((base == 0) & (y == 1)); broke = ((base == 1) & (y == 0))
    print(f"  {name:<46s} c={c[-3:]:>4s} | whole data {100 * acc(steer_mask, c):5.2f}% | facts 1-4/5 {100 * acc(steer_mask, c, mid):5.2f}% | steered: whole {100 * (steer_mask * w).sum() / w.sum():5.1f}%, 1-4/5 {100 * steer_mask[mid].mean():5.1f}% | in 1-4/5: fixed {int(fixed[mid].sum())}, broken {int(broke[mid].sum())}")
none = np.zeros(len(idx), bool); allm = np.ones(len(idx), bool)
print(f"rows: 1-4/5 {mid.sum()}, 0/5 sample {(k == 0).sum()}, 5/5 sample {(k == 5).sum()}")
print(f"base accuracy: whole data {100 * acc(none, 'B c1'):.2f}% | facts 1-4/5 {100 * acc(none, 'B c1', mid):.2f}%   (0/5 sample {100 * base[k == 0].mean():.1f}%, 5/5 sample {100 * base[k == 5].mean():.1f}%)")
print("\n[no detector]")
for c in ("B c0.5", "B c1"): report("steer every row", allm, c)
for c in ("B c0.5", "B c1"): report("steer only facts 1/5-4/5 (perfect knowledge gate)", mid, c)
print("\n[oracle: knows which rows are wrong at base]")
for c in ("B c0.5", "B c1"): report("steer only rows that are wrong", base == 0, c)
for c in ("B c0.5", "B c1"): report("steer only wrong rows of facts 1/5-4/5", (base == 0) & mid, c)
print("\n[detectors, threshold and c chosen on the other 4 folds]")
qs = np.linspace(0.02, 0.98, 49)
def nested(rule_name, make_mask):
    mask = np.zeros(len(idx), bool); cs = np.empty(len(idx), object); chosen = []
    for g in range(5):
        tr, te = f != g, f == g; best = (-1, None, None)
        for c in ("B c0.5", "B c1"):
            for par in make_mask.grid:
                m = make_mask(par)
                y = np.where(m, st[c], base); a = (y[tr] * w[tr]).sum() / w[tr].sum()
                if a > best[0]: best = (a, c, par)
        _, c, par = best; m = make_mask(par); mask[te] = m[te]; cs[te] = c; chosen.append((c, (qs[par[0]], qs[par[1]]) if isinstance(par, tuple) else qs[par]))
    y = np.where(mask, np.array([st[c][j] for j, c in enumerate(cs)]), base)
    fixed = (base == 0) & (y == 1); broke = (base == 1) & (y == 0)
    a_w = (y * w).sum() / w.sum(); a_m = y[mid].mean()
    print(f"  {rule_name:<46s} | whole data {100 * a_w:5.2f}% | facts 1-4/5 {100 * a_m:5.2f}% | steered: whole {100 * (mask * w).sum() / w.sum():5.1f}%, 1-4/5 {100 * mask[mid].mean():5.1f}% | in 1-4/5: fixed {int(fixed[mid].sum())}, broken {int(broke[mid].sum())}")
    print(f"      chosen per fold (c, threshold quantile): {[(c[-3:], tuple(np.round(p, 2)) if isinstance(p, tuple) else round(float(p), 2)) for c, p in chosen]}")
for n in ("P_all", "P_mid", "axis"):
    s = sc[n]; qv = np.quantile(s, qs)
    mm = lambda p, s=s, qv=qv: s < qv[p]; mm.grid = range(len(qs)); nested(f"steer if {n} is low (likely wrong)", mm)
pk, pm = sc["P_know"], sc["P_mid"]; qk, qm = np.quantile(pk, qs), np.quantile(pm, qs)
grid = [(a, b) for a in range(0, len(qs), 4) for b in range(0, len(qs), 4)]
mm = lambda p: (pk > qk[p[0]]) & (pm < qm[p[1]]); mm.grid = grid; nested("steer if P_know high AND P_mid low", mm)
