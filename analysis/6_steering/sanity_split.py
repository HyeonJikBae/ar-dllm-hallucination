"""Sanity checks for steer_from_0of5.py: are the groups and the signs what we think?
1) who are the target rows (k, stored type, rule)   2) rule mix of broken rows in 0/5 vs 1/5..4/5
3) firing rates of the 20 BROKEN-side and 20 ENTITY-side latents in each group (0/5 broken, 0/5 entity, and 1/5..4/5 correct / entity / broken)."""
import collections, json, sys
import numpy as np, torch
sys.path.insert(0, "analysis"); import sae_io
D = "data"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); rule = np.array([r["rule"] for r in rows]); mid = (k >= 1) & (k <= 4)
tgt = np.where(mid & (ty == "broken"))[0]
print("targets:", len(tgt), "| k:", dict(collections.Counter(k[tgt].tolist())), "| stored type:", dict(collections.Counter(ty[tgt].tolist())))
for name, m in (("broken 0/5", (k == 0) & (ty == "broken")), ("broken 1-4/5", mid & (ty == "broken"))):
    c = collections.Counter(rule[m]); n = m.sum(); print(f"rule mix {name} (n={n}):", {r_: round(v / n, 3) for r_, v in c.most_common()})
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, True); H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
def onm(idx):
    o = np.zeros((len(idx), 16384), bool)
    for b in range(0, len(idx), 1024): o[b:b + 1024] = (enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in idx[b:b + 1024]])).cuda()) > 0).cpu().numpy()
    return o
z = np.where((k == 0) & np.isin(ty, ["broken", "entity"]))[0]; Oz = onm(z); tz, fz = ty[z], fid[z]
byf = collections.defaultdict(lambda: collections.defaultdict(list))
for j, (f, t) in enumerate(zip(fz, tz)): byf[f][t].append(j)
sc = np.mean([Oz[v["broken"]].mean(0) - Oz[v["entity"]].mean(0) for v in byf.values() if v["broken"] and v["entity"]], 0)
allrate = Oz.mean(0); rb, re_ = Oz[tz == "broken"].mean(0), Oz[tz == "entity"].mean(0)
dn = np.argsort(-np.where((allrate <= .5) & (rb >= .05), sc, -9))[:20]; up = np.argsort(-np.where((allrate <= .5) & (re_ >= .05), -sc, -9))[:20]
mi = np.where(mid & np.isin(ty, ["correct", "entity", "broken"]))[0]; Om = onm(mi); tm = ty[mi]
print("\nshare of (row, latent) pairs ON:             BROKEN-side 20 | ENTITY-side 20")
for name, O, sel in (("0/5 broken", Oz, tz == "broken"), ("0/5 entity", Oz, tz == "entity"), ("1-4/5 correct", Om, tm == "correct"), ("1-4/5 entity", Om, tm == "entity"), ("1-4/5 broken (targets)", Om, tm == "broken")):
    print(f"  {name:<24s} n={int(sel.sum()):>6d}        {O[sel][:, dn].mean():.3f}          | {O[sel][:, up].mean():.3f}")
