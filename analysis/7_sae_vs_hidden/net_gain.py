"""Net gain if the edit is applied to EVERY row (correct and wrong), from results_caa_layers.jsonl.  CPU.
The sample has 300 wrong and 300 correct rows of facts 1/5..4/5; they are weighted to the real mix of facts 1/5..4/5 (52.4% wrong, 47.6% correct):
  net = 0.524 * (share of wrong rows made correct) - 0.476 * (share of correct rows made wrong),   both measured against the no-edit run of the same rows.
Also: accuracy if only the wrong rows were edited (oracle gate).   95% intervals: bootstrap over rows (2,000 resamples, wrong and correct rows resampled separately).
Usage (repo root): python3 analysis/7_sae_vs_hidden/net_gain.py"""
import collections, json
import numpy as np
PW, PC = 0.524, 0.476
R = collections.defaultdict(dict); was = {}
for l in open("analysis/7_sae_vs_hidden/results_caa_layers.jsonl", encoding="utf-8"):
    d = json.loads(l); R[d["cond"]][d["row"]] = d["correct"]; was[d["row"]] = d["was_correct"]
rows = np.array(sorted(was)); w = np.array([was[r] for r in rows]); iw, ic = np.where(~w)[0], np.where(w)[0]
base = np.array([R["base"][r] for r in rows]); rng = np.random.default_rng(0)
def net(s, a, b): return PW * (s[a].mean() - base[a].mean()) + PC * (s[b].mean() - base[b].mean())
print(f"rows: wrong {len(iw)}, correct {len(ic)} | no edit: wrong rows correct {base[iw].mean():.3f}, correct rows correct {base[ic].mean():.3f}")
print(f"{'condition':<12s} wrong->correct  correct kept | net gain (all rows edited)  95% interval   | only wrong rows edited: gain")
for c in R:
    if c == "base": continue
    s = np.array([R[c][r] for r in rows]); g = net(s, iw, ic)
    bs = [net(s, rng.choice(iw, len(iw)), rng.choice(ic, len(ic))) for _ in range(2000)]
    print(f"{c:<12s} {s[iw].mean():.3f}           {s[ic].mean():.3f}        | {g * 100:+5.1f} pt                  [{np.percentile(bs, 2.5) * 100:+5.1f}, {np.percentile(bs, 97.5) * 100:+5.1f}] "
          f"| {PW * (s[iw].mean() - base[iw].mean()) * 100:+5.1f} pt")
