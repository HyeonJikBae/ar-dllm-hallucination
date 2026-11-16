"""Why can removing RANDOM latents reduce broken more than removing the broken-signal latents?  Test: the size of the edit.
Rebuilds the exact latent sets of steer_from_0of5.py (same selection, same random draws), and for each removal condition measures, on the saved step-0 prompt-end states of the 310 target rows,
the share of ||h||^2 that the projection removes.  Then compares it with the broken rate that condition produced (results_from0of5.jsonl).
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/6_steering/removed_energy.py"""
import collections, json, sys
import numpy as np, torch
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, "analysis"); import sae_io
D = "data"
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k_all = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k_all >= 1) & (k_all <= 4)
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
Wu = Wd["W_dec"] / Wd["W_dec"].norm(dim=1, keepdim=True); H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
z = np.where((k_all == 0) & np.isin(ty, ["broken", "entity"]))[0]; Oz = np.zeros((len(z), 16384), np.bool_)
for b in range(0, len(z), 1024): Oz[b:b + 1024] = (enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in z[b:b + 1024]])).cuda()) > 0).cpu().numpy()
tz, fz = ty[z], fid[z]; byf = collections.defaultdict(lambda: collections.defaultdict(list))
for j, (f, t) in enumerate(zip(fz, tz)): byf[f][t].append(j)
sc = np.mean([Oz[v["broken"]].mean(0) - Oz[v["entity"]].mean(0) for v in byf.values() if v["broken"] and v["entity"]], 0)
allrate = Oz.mean(0); rb, re_ = Oz[tz == "broken"].mean(0), Oz[tz == "entity"].mean(0); okb, oke = (allrate <= .5) & (rb >= .05), (allrate <= .5) & (re_ >= .05)
sides = lambda K: (np.argsort(-np.where(okb, sc, -9))[:K], np.argsort(-np.where(oke, -sc, -9))[:K]); dn20, up20 = sides(20); dn5, _ = sides(5)
cand = np.where(allrate > .01)[0]; qs = np.quantile(allrate[cand], np.linspace(0, 1, 11)); dec = np.clip(np.digitize(allrate, qs[1:-1]), 0, 9); rr = np.random.default_rng(5); excl = set(dn20) | set(up20)
def draw(lat):
    pl = []
    for j in lat: pl.append(int(rr.choice([c for c in cand if dec[c] == dec[j] and c not in excl and c not in pl])))
    return pl
sets = {"remove BROKEN K20": dn20, "remove ENTITY-side K20": up20, "remove BROKEN K5": dn5}
sets.update({f"remove random 20 #{q}": draw(dn20) for q in range(8)}); sets.update({f"remove random 5 #{q}": draw(dn5) for q in range(8)})
tgt = np.where(mid & (ty == "broken"))[0]; X = torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in tgt]))
res = collections.defaultdict(dict)
for l in open("analysis/6_steering/results_from0of5.jsonl", encoding="utf-8"): r = json.loads(l); res[r["cond"]][r["row"]] = r["new_type"]
out = []
for name, lat in sets.items():
    Q = torch.linalg.qr(Wu[lat].T.float())[0]; rem = ((X @ Q).norm(dim=1) ** 2 / X.norm(dim=1) ** 2).mean().item()
    on_t = (enc(X.cuda()).cpu().numpy()[:, lat] > 0).mean()
    br = np.mean([res[name][int(i)] == "broken" for i in tgt]); out.append((name, rem, br, on_t))
print(f"{'condition':<26s} share of ||h||^2 removed | latents ON in target rows | broken after")
for n, rem, br, o in out: print(f"  {n:<26s} {rem:.4f}                    {o:.3f}                       {br:.3f}")
for grp in ("20", "5"):
    sel = [(rem, br) for n, rem, br, _ in out if (n.endswith(f"K{grp}") or f"random {grp} " in n)]
    a, b = np.array(sel).T; print(f"K={grp}: over {len(a)} conditions, correlation(removed share, broken after): Pearson {pearsonr(a, b)[0]:.2f}, Spearman {spearmanr(a, b)[0]:.2f}")
