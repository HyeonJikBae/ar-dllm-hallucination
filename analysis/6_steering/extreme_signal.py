"""Signals found on the EXTREME groups (no 1/5..4/5 row used), then tested on facts 1/5..4/5.  CPU only.  Dream, SAE layer 20, prompt end, step 0.
 signal 1 "5/5 correct vs 0/5 broken":      latents firing more in correct rows of 5/5 facts / in broken rows of 0/5 facts   (confounded with knowing the fact)
 signal 2 "0/5 broken vs 0/5 entity":       latents firing more in broken rows / in entity-wrong rows, both from facts the model never gets right (knowledge held constant)
 signal 1b "5/5 correct vs 0/5 entity":   the same with entity-wrong rows of 0/5 facts (a 5/5 fact has no wrong row, so its wrong-answer partner must come from 0/5 facts)
 signal 3 "within-fact, all 1/5..4/5 rows":  the MaxMin signal of crossfit_broken.py found on all 1/5..4/5 rows  (in-sample reference)
Top-20 per side (separation score of ON/OFF).  Test on facts 1/5..4/5: within-fact AUROC of (#broken-side latents ON - #correct-side latents ON) for correct-vs-broken and entity-vs-broken, and overlap with signal 3."""
import collections, json, sys, warnings
import numpy as np, torch
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis"); torch.set_num_threads(16)
import sae_io
D = "data"; K = 20
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
rng = np.random.default_rng(0)
g_c5 = np.where((ty == "correct") & (k == 5))[0]; g_b0 = np.where((ty == "broken") & (k == 0))[0]; g_e0 = rng.permutation(np.where((ty == "entity") & (k == 0))[0])[:6000]
g_mid = np.where(mid & np.isin(ty, ["correct", "broken", "entity"]))[0]
idx = np.unique(np.concatenate([g_c5, g_b0, g_e0, g_mid])); print("rows to encode", len(idx), {"5/5 correct": len(g_c5), "0/5 broken": len(g_b0), "0/5 entity (sample)": len(g_e0), "1-4/5 rows": len(g_mid)}, flush=True)
enc = sae_io.dream_sae(); H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
on = np.zeros((len(idx), 16384), np.float32)
for b in range(0, len(idx), 1024): on[b:b + 1024] = (enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in idx[b:b + 1024]]))).numpy() > 0)
pos = {int(i): j for j, i in enumerate(idx)}; P = lambda g: np.array([pos[int(i)] for i in g]); allrate = on.mean(0)
sep = lambda a, b: on[P(a)].mean(0) - on[P(b)].mean(0)
def top(score, ok): return np.argsort(-np.where(ok, score, -9))[:K]
ok = allrate <= 0.5
s1 = sep(g_c5, g_b0); s1b = sep(g_c5, g_e0); s2 = sep(g_e0, g_b0)                                          # >0 : fires more in correct (1) / entity (2) than in broken
sig = {"1: 5/5 correct vs 0/5 broken": (top(s1, ok & (on[P(g_c5)].mean(0) >= .05)), top(-s1, ok & (on[P(g_b0)].mean(0) >= .05))),
       "1b: 5/5 correct vs 0/5 entity": (top(s1b, ok & (on[P(g_c5)].mean(0) >= .05)), top(-s1b, ok & (on[P(g_e0)].mean(0) >= .05))),
       "2: 0/5 broken vs 0/5 entity": (top(s2, ok & (on[P(g_e0)].mean(0) >= .05)), top(-s2, ok & (on[P(g_b0)].mean(0) >= .05)))}
byf = collections.defaultdict(lambda: {"i": collections.defaultdict(list), "k": 0})
for i in g_mid: byf[fid[i]]["i"][ty[i]].append(pos[int(i)]); byf[fid[i]]["k"] = k[i]
rate = lambda f, t: on[byf[f]["i"][t]].mean(0)
d = {g: [] for g in (1, 2, 3, 4)}; e, ce = [], []
for f, v in byf.items():
    if v["i"]["correct"] and v["i"]["broken"]: d[v["k"]].append(rate(f, "correct") - rate(f, "broken"))
    if v["i"]["entity"] and v["i"]["broken"]: e.append(rate(f, "entity") - rate(f, "broken"))
    if v["i"]["correct"] and v["i"]["entity"]: ce.append(rate(f, "correct") - rate(f, "entity"))
sc_c = np.min([np.mean(d[g], 0) for g in d if d[g]] + [np.mean(ce, 0)], 0); sc_b = np.min([-np.mean(d[g], 0) for g in d if d[g]] + [-np.mean(e, 0)], 0)
midrate = on[P(g_mid)].mean(0); okm = midrate <= 0.5
sig["3: within-fact 1/5-4/5 (in-sample)"] = (top(sc_c, okm & (on[P(g_mid[ty[g_mid] == "correct"])].mean(0) >= .05)), top(sc_b, okm & (on[P(g_mid[ty[g_mid] == "broken"])].mean(0) >= .05)))
print("\nwithin-fact AUROC on facts 1/5..4/5 of  score = (#broken-side latents ON) - (#correct-side latents ON), top-20 each")
for name, (up, dn) in sig.items():
    s = np.zeros(len(g_mid)); X = on[P(g_mid)]; s = X[:, dn].sum(1) - X[:, up].sum(1); t = ty[g_mid]; f = fid[g_mid]; out = []
    for a_, b_ in (("correct", "broken"), ("entity", "broken"), ("correct", "entity")):
        per = []
        for u in np.unique(f):
            m = (f == u) & np.isin(t, [a_, b_])
            if len(set(t[m])) == 2: per.append(roc_auc_score((t[m] == b_).astype(int), s[m]))
        out.append(f"{a_}-{b_} {np.mean(per):.3f} ({len(per)} facts)")
    print(f"  {name:<38s} " + " | ".join(out))
print('overlap of the top-20 WRONG-side latents between signal 1 (broken) and 1b (entity):', len(set(sig['1: 5/5 correct vs 0/5 broken'][1]) & set(sig['1b: 5/5 correct vs 0/5 entity'][1])), '| correct side:', len(set(sig['1: 5/5 correct vs 0/5 broken'][0]) & set(sig['1b: 5/5 correct vs 0/5 entity'][0])))
print("\noverlap of the top-20 BROKEN-side latents with signal 3:", {n: len(set(v[1]) & set(sig['3: within-fact 1/5-4/5 (in-sample)'][1])) for n, v in sig.items() if not n.startswith('3')})
print("overlap of the top-20 CORRECT-side latents with signal 3:", {n: len(set(v[0]) & set(sig['3: within-fact 1/5-4/5 (in-sample)'][0])) for n, v in sig.items() if not n.startswith('3')})
print("overlap between signal 1 and 2 (broken side):", len(set(sig['1: 5/5 correct vs 0/5 broken'][1]) & set(sig['2: 0/5 broken vs 0/5 entity'][1])))
