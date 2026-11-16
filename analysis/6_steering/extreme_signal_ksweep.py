"""K SWEEP (K = 1, 10, 20, 100 latents per side; ON/OFF and strength) of extreme_signal.py.

Signals found on the EXTREME groups (no 1/5..4/5 row used), then tested on facts 1/5..4/5.  CPU only.  Dream, SAE layer 20, prompt end, step 0.
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
A = np.zeros((len(idx), 16384), np.float16)
for b in range(0, len(idx), 1024): A[b:b + 1024] = enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in idx[b:b + 1024]]))).numpy().astype(np.float16)
pos = {int(i): j for j, i in enumerate(idx)}; P = lambda g: np.array([pos[int(i)] for i in g]); on = (A > 0).astype(np.float32); allrate = on.mean(0); ok = allrate <= 0.5
def stats(a, b):
    Xa, Xb = A[P(a)].astype(np.float32), A[P(b)].astype(np.float32)
    return on[P(a)].mean(0) - on[P(b)].mean(0), (Xa.mean(0) - Xb.mean(0)) / np.sqrt(Xa.var(0) / len(Xa) + Xb.var(0) / len(Xb) + 1e-8), on[P(a)].mean(0), on[P(b)].mean(0)
st1, st1b, st2 = stats(g_c5, g_b0), stats(g_c5, g_e0), stats(g_e0, g_b0)           # (separation, t, rate in first group, rate in second group); first group = "correct / entity side"
byf = collections.defaultdict(lambda: {"i": collections.defaultdict(list), "k": 0})
for i in g_mid: byf[fid[i]]["i"][ty[i]].append(pos[int(i)]); byf[fid[i]]["k"] = k[i]
rate = lambda f, t: on[byf[f]["i"][t]].mean(0)
d = {g: [] for g in (1, 2, 3, 4)}; e, ce = [], []
for f, v in byf.items():
    if v["i"]["correct"] and v["i"]["broken"]: d[v["k"]].append(rate(f, "correct") - rate(f, "broken"))
    if v["i"]["entity"] and v["i"]["broken"]: e.append(rate(f, "entity") - rate(f, "broken"))
    if v["i"]["correct"] and v["i"]["entity"]: ce.append(rate(f, "correct") - rate(f, "entity"))
sc_c = np.min([np.mean(d[g], 0) for g in d if d[g]] + [np.mean(ce, 0)], 0); sc_b = np.min([-np.mean(d[g], 0) for g in d if d[g]] + [-np.mean(e, 0)], 0)
rc, rb_ = on[P(g_mid[ty[g_mid] == "correct"])].mean(0), on[P(g_mid[ty[g_mid] == "broken"])].mean(0); midok = on[P(g_mid)].mean(0) <= 0.5
Xm = A[P(g_mid)].astype(np.float32); Om = (Xm > 0).astype(np.float32); mu, sd = Xm.mean(0), Xm.std(0) + 1e-6; Zm = (Xm - mu) / sd; t_, f_ = ty[g_mid], fid[g_mid]
def lists(score, ra, rb, K):                                                         # score high = first-group side
    return np.argsort(-np.where(ok & (ra >= .05), score, -9))[:K], np.argsort(-np.where(ok & (rb >= .05), -score, -9))[:K]
def within(s, a_, b_):
    per = []
    for u in np.unique(f_):
        m = (f_ == u) & np.isin(t_, [a_, b_])
        if len(set(t_[m])) == 2: per.append(roc_auc_score((t_[m] == b_).astype(int), s[m]))
    return np.mean(per)
SIGS = {"1: 5/5 correct vs 0/5 broken": (st1, "sep", "t"), "1b: 5/5 correct vs 0/5 entity": (st1b, "sep", "t"), "2: 0/5 broken vs 0/5 entity": (st2, "sep", "t")}
print("within-fact AUROC on facts 1/5..4/5 (cb = correct-vs-broken 213 facts, eb = entity-vs-broken 142, ce = correct-vs-entity 1118);  score = broken-side minus correct-side")
for name, (st, _, _) in SIGS.items():
    for kind, ix in (("ON/OFF", 0), ("strength", 1)):
        for K in (1, 10, 20, 100):
            up, dn = lists(st[ix], st[2], st[3], K); s = Om[:, dn].sum(1) - Om[:, up].sum(1) if ix == 0 else Zm[:, dn].sum(1) - Zm[:, up].sum(1)
            print(f"  {name:<32s} {kind:<8s} K={K:<3d} cb {within(s, 'correct', 'broken'):.3f}  eb {within(s, 'entity', 'broken'):.3f}  ce {within(s, 'correct', 'entity'):.3f}")
for kind in ("ON/OFF",):
    for K in (1, 10, 20, 100):
        up = np.argsort(-np.where(midok & (rc >= .05), sc_c, -9))[:K]; dn = np.argsort(-np.where(midok & (rb_ >= .05), sc_b, -9))[:K]; s = Om[:, dn].sum(1) - Om[:, up].sum(1)
        print(f"  {'3: within-fact 1/5-4/5 (in-sample)':<32s} {kind:<8s} K={K:<3d} cb {within(s, 'correct', 'broken'):.3f}  eb {within(s, 'entity', 'broken'):.3f}  ce {within(s, 'correct', 'entity'):.3f}")
