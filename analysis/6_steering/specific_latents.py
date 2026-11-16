"""Remove the "everyone has it" part: pick broken latents by how SPECIFIC they are, not by the raw difference in firing rate.  0/5 facts only (as steer_from_0of5.py), tested on facts 1/5..4/5.
Selections (top 20 broken-side; the opposite side is chosen the same way for entity):
  diff     : (rate in broken) - (rate in entity), within 0/5 facts      [the current choice; latents firing on > 50% of rows dropped]
  ratio    : log((rate broken + 0.01) / (rate entity + 0.01)), latents firing in >= 5% of broken rows and <= 20% of all 0/5 rows
  ratio10  : the same with <= 10% of all rows
  centred  : diff computed after subtracting each latent's mean firing over ALL 0/5 rows from every row (the shared part), then divided by its spread (z-score of the rate gap)
For each: average share ON in each group, and within-fact AUROC on facts 1/5..4/5 of (#broken-side ON - #entity-side ON)."""
import collections, json, sys
import numpy as np, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "analysis"); import sae_io
D = "data"; K = 20
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, True); H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
def onm(idx):
    o = np.zeros((len(idx), 16384), bool)
    for b in range(0, len(idx), 1024): o[b:b + 1024] = (enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in idx[b:b + 1024]])).cuda()) > 0).cpu().numpy()
    return o
z = np.where((k == 0) & np.isin(ty, ["broken", "entity"]))[0]; Oz = onm(z); tz, fz = ty[z], fid[z]
byf = collections.defaultdict(lambda: collections.defaultdict(list))
for j, (f, t) in enumerate(zip(fz, tz)): byf[f][t].append(j)
pairs = [(v["broken"], v["entity"]) for v in byf.values() if v["broken"] and v["entity"]]
allrate = Oz.mean(0); rb, re_ = Oz[tz == "broken"].mean(0), Oz[tz == "entity"].mean(0)
diff = np.mean([Oz[b].mean(0) - Oz[e].mean(0) for b, e in pairs], 0)
lr = np.log((rb + .01) / (re_ + .01))
gap = np.array([Oz[b].mean(0) - Oz[e].mean(0) for b, e in pairs]); cz = gap.mean(0) / (gap.std(0) / np.sqrt(len(gap)) + 1e-6)   # within-fact gap in units of its standard error
SEL = {"diff": (diff, (allrate <= .5) & (rb >= .05), (allrate <= .5) & (re_ >= .05)),
       "ratio (<=20% of rows)": (lr, (allrate <= .2) & (rb >= .05), (allrate <= .2) & (re_ >= .05)),
       "ratio (<=10% of rows)": (lr, (allrate <= .1) & (rb >= .05), (allrate <= .1) & (re_ >= .05)),
       "centred z of the gap": (cz, (allrate <= .5) & (rb >= .05), (allrate <= .5) & (re_ >= .05))}
mi = np.where(mid & np.isin(ty, ["correct", "entity", "broken"]))[0]; Om = onm(mi); tm, fm = ty[mi], fid[mi]
def within(s, a_, b_):
    per = [roc_auc_score((tm[m] == b_).astype(int), s[m]) for u in np.unique(fm) for m in [(fm == u) & np.isin(tm, [a_, b_])] if len(set(tm[m])) == 2]; return np.mean(per)
print(f"{'selection':<24s} | share ON of the 20 BROKEN-side latents: 0/5 brk, 0/5 ent, 1-4/5 cor, 1-4/5 ent, 1-4/5 brk | lift brk/ent (0/5) | within-fact AUROC 1-4/5: cor-brk, ent-brk, cor-ent")
for name, (score, okb, oke) in SEL.items():
    dn = np.argsort(-np.where(okb, score, -1e9))[:K]; up = np.argsort(-np.where(oke, -score, -1e9))[:K]
    r = [Oz[tz == "broken"][:, dn].mean(), Oz[tz == "entity"][:, dn].mean(), Om[tm == "correct"][:, dn].mean(), Om[tm == "entity"][:, dn].mean(), Om[tm == "broken"][:, dn].mean()]
    s = Om[:, dn].sum(1).astype(float) - Om[:, up].sum(1)
    print(f"{name:<24s} | {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f} {r[4]:.3f} | {r[0] / max(r[1], 1e-6):.2f} | {within(s, 'correct', 'broken'):.3f} {within(s, 'entity', 'broken'):.3f} {within(s, 'correct', 'entity'):.3f}")
