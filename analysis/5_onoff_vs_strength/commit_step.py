"""At which denoising step does the answer get fixed?  (Dream; a slot is never re-masked once filled.)
For each row: first step at which each answer slot is no longer masked (conf_mask[row, step, slot] = still masked at that step).
Answer slots = the slots from the first real (non-space, non-line-break) token up to the first special token.  Reported per fact group x correct/wrong, for facts 1/5..4/5 too.
Usage: python3 analysis/5_onoff_vs_strength/commit_step.py"""
import json, numpy as np
from transformers import AutoTokenizer
D = "data"; F = f"{D}/1_raw/dream_hidden_steps"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows]); ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows])
T = np.load(f"{F}/tokens.npy"); M = np.load(f"{F}/conf_mask.npy", mmap_mode="r")
tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True); SP = set(tok.all_special_ids)
WS = {int(i) for i in np.unique(T) if i >= 0 and int(i) not in SP and tok.decode([int(i)]).strip() == ""}   # tokens that are only spaces / line breaks
first, last, mn = np.full(len(rows), -1), np.full(len(rows), -1), np.full(len(rows), -1)
for b in range(0, len(rows), 4096):
    m = np.asarray(M[b:b + 4096])                         # [n, 32 step, 32 slot]
    still = m.astype(np.int8)
    # first step where slot unmasked = number of leading masked steps
    commit = (np.cumprod(still, axis=1)).sum(1)            # [n, slot]  (steps spent masked)
    for j in range(len(m)):
        t = T[b + j]; n = next((i for i, x in enumerate(t) if int(x) in SP), 32)
        n0 = 0                                           # skip leading space / line-break tokens: the first real answer token
        while n0 < n and int(t[n0]) in WS: n0 += 1
        if n0 >= n: continue
        c = commit[j, n0:n]; first[b + j] = c[0]; last[b + j] = c.max(); mn[b + j] = c.min()
print("steps spent masked before the slot was filled (0 = filled at the first step, 31 = last)\n")
sel = (first >= 0)
def row(name, m):
    m = m & sel
    if m.sum() == 0: return
    print(f"{name:<28s} n={m.sum():>6d} | answer slot 0 filled by: median {np.median(first[m]):.0f}, mean {first[m].mean():.1f} | whole answer done: median {np.median(last[m]):.0f}, mean {last[m].mean():.1f} | slot0 filled within the first 8 steps: {(first[m] < 8).mean():.2f}")
row("correct", cor); row("wrong (all)", ~cor)
for t in ("entity", "broken", "abstain"): row(f"wrong {t}", ty == t)
mid = (k >= 1) & (k <= 4)
print("\nfacts 1/5..4/5 only"); row("correct", mid & cor); row("wrong", mid & ~cor)
print("\nshare of answer slot 0 filled by step (cumulative), correct vs wrong")
for s in (0, 1, 2, 4, 8, 16, 24, 31):
    print(f"  step<={s:>2d}  correct {(first[sel & cor] <= s).mean():.2f}   wrong {(first[sel & ~cor] <= s).mean():.2f}")
