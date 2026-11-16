"""How similar are the correct / broken signals found on different data?  (CPU only)  Dream, SAE layer 20, prompt-end token step 0, facts 1/5..4/5.
Signals: the cross-fit MaxMin within-fact signal (crossfit_broken.py) found on half 1, on half 2, and on ALL rows;  the earlier pooled t-statistic signal (steer_broken.py / ablate_broken.py, found on half 1 rows of correct vs broken).
Reports the overlap of the top-20 latents of each kind and the rank correlation of the latent scores."""
import collections, json, sys, warnings
import numpy as np, torch
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis"); torch.set_num_threads(16)
import sae_io
D = "data"; K = 20
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); half = np.isin(sub, subs[: len(subs) // 2])
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k_all = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k_all >= 1) & (k_all <= 4)
enc = sae_io.dream_sae(); H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
idx_all = np.where(mid & np.isin(ty, ["correct", "broken", "entity"]))[0]
A = np.concatenate([enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in idx_all[b:b + 512]]))).numpy() for b in range(0, len(idx_all), 512)]); pos = {int(i): j for j, i in enumerate(idx_all)}
on = (A > 0).astype(np.float32)
def maxmin(src):
    sel = [i for i in idx_all if src[i]]; byf = collections.defaultdict(lambda: {"idx": collections.defaultdict(list), "k": 0})
    for i in sel: byf[fid[i]]["idx"][ty[i]].append(pos[int(i)]); byf[fid[i]]["k"] = k_all[i]
    rate = lambda f, t: on[byf[f]["idx"][t]].mean(0)
    d = {g: [] for g in (1, 2, 3, 4)}; e, ce = [], []
    for f, v in byf.items():
        if v["idx"]["correct"] and v["idx"]["broken"]: d[v["k"]].append(rate(f, "correct") - rate(f, "broken"))
        if v["idx"]["entity"] and v["idx"]["broken"]: e.append(rate(f, "entity") - rate(f, "broken"))
        if v["idx"]["correct"] and v["idx"]["entity"]: ce.append(rate(f, "correct") - rate(f, "entity"))
    sc_c = np.min([np.mean(d[g], 0) for g in d if d[g]] + [np.mean(ce, 0)], 0); sc_b = np.min([-np.mean(d[g], 0) for g in d if d[g]] + [-np.mean(e, 0)], 0)
    rows_ = [pos[int(i)] for i in sel]; t = np.array([ty[i] for i in sel]); allrate = on[rows_].mean(0)
    okc = (allrate <= 0.5) & (on[np.array(rows_)[t == "correct"]].mean(0) >= 0.05); okb = (allrate <= 0.5) & (on[np.array(rows_)[t == "broken"]].mean(0) >= 0.05)
    return np.where(okc, sc_c, -9), np.where(okb, sc_b, -9)
def tstat(src):                                                                     # the earlier pooled t-statistic signal
    sel = np.array([i for i in idx_all if src[i] and ty[i] in ("correct", "broken")]); a = A[[pos[int(i)] for i in sel]]; y = np.array([ty[i] == "correct" for i in sel])
    P, Q = a[y], a[~y]; t = (P.mean(0) - Q.mean(0)) / np.sqrt(P.var(0) / len(P) + Q.var(0) / len(Q) + 1e-8); return t, -t
S = {"cross-fit half 1": maxmin(half), "cross-fit half 2": maxmin(~half), "cross-fit ALL rows": maxmin(np.ones(len(rows), bool)), "earlier t-stat (half 1)": tstat(half), "earlier t-stat (ALL rows)": tstat(np.ones(len(rows), bool))}
top = lambda v: set(np.argsort(-v)[:K].tolist()); names = list(S)
for kind, ix in (("BROKEN signal", 1), ("CORRECT signal", 0)):
    print(f"\n=== {kind}: overlap of the top-{K} latents (lower triangle) and Spearman correlation of all latent scores over latents with a score (upper triangle)")
    print(" " * 26 + "  ".join(f"{i}" for i in range(len(names))))
    for a, na in enumerate(names):
        cells = []
        for b, nb in enumerate(names):
            if a == b: cells.append(" - ")
            elif b < a: cells.append(f"{len(top(S[na][ix]) & top(S[nb][ix])):>3d}")
            else: ok = (S[na][ix] > -9) & (S[nb][ix] > -9); cells.append(f"{spearmanr(S[na][ix][ok], S[nb][ix][ok])[0]:.2f}")
        print(f"{a} {na:<24s}" + "  ".join(cells))
