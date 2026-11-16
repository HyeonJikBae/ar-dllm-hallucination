"""Broken signal found ONLY on facts the model never gets right (0/5): broken rows vs entity-wrong rows of the same 0/5 fact (knowledge held constant), then steering on the broken rows of facts 1/5..4/5 (never used for the signal, so no leakage).
Dream-v0-Base-7B, layer 20, prompt-end token, SAE (16k), step-0 states.
Signal: per 0/5 fact with both kinds of rows, (firing rate in broken rows) - (firing rate in entity rows), averaged over facts.  Latents firing on > 50% of all 0/5 rows are dropped; a signal latent must fire in >= 5% of the rows of its own kind.
  BROKEN-side latents = top K (fire more in broken rows);  ENTITY-side latents = top K (fire more in entity rows).   K = 20, plus K = 5 for the removal test.
Check: within-fact AUROC on facts 1/5..4/5 of (#broken-side ON - #entity-side ON):  correct vs broken, entity vs broken, correct vs entity.
Edits at the prompt-end token's layer-20 output in the first T=8 denoising forwards:  remove = project out the span of the K unit decoder directions;  add = + alpha * mean||h|| * unit(sum of the K unit decoder directions).
Conditions: base | base batch 8 (noise) | remove BROKEN (K20) | add BROKEN a0.5, a1 | remove ENTITY-side (K20) | add ENTITY-side a0.5, a1 | remove BROKEN (K5)
  controls: remove 20 random latents x8 | remove 5 random latents x8 | add 20 random latents a1 x3   (random latents matched to the signal latents by firing-rate decile)
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/6_steering/steer_from_0of5.py [--pilot 8] [--draws 8]"""
import argparse, collections, json, os, sys, time
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "analysis"); sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import sae_io, wd
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
from classify_rules import classify
ap = argparse.ArgumentParser(); ap.add_argument("--pilot", type=int, default=0); ap.add_argument("--draws", type=int, default=8); ap.add_argument("--add-draws", type=int, default=3)
ap.add_argument("--match-size", action="store_true", help="random removal sets must remove about the same share of ||h||^2 (on the target rows, saved step-0 states) as the signal set: within +-15%"); ap.add_argument("--select", choices=["diff", "ratio"], default="diff", help="diff: rate gap (latents firing on <= 50%% of 0/5 rows); ratio: log((rate broken+.01)/(rate entity+.01)), latents firing on <= 20%% of 0/5 rows (specific latents)"); ap.add_argument("--remove-mode", choices=["proj", "sae"], default="proj", help="proj: project out the span of the decoder directions (changes every row); sae: subtract each latent's own SAE contribution a_j(h) * W_dec[j] (changes a row only where that latent is ON)"); ap.add_argument("--lean", action="store_true", help="only base, remove BROKEN K20/K5, remove ENTITY-side K20, add BROKEN a1, add ENTITY-side a1, and the random controls"); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--T", type=int, default=8); ap.add_argument("--out", default="analysis/6_steering/results_from0of5.jsonl")
args = ap.parse_args()
D = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS = 20, 32, 32
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k_all = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k_all >= 1) & (k_all <= 4)
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
Wu = Wd["W_dec"] / Wd["W_dec"].norm(dim=1, keepdim=True); Wg = {a: b.cuda() for a, b in Wd.items()}; SUB_BDEC = json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True); H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
def onmat(idx):
    out = np.zeros((len(idx), 16384), np.bool_)
    for b in range(0, len(idx), 1024): out[b:b + 1024] = (enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in idx[b:b + 1024]])).cuda()) > 0).cpu().numpy()
    return out
# ---- signal on 0/5 facts
z = np.where((k_all == 0) & np.isin(ty, ["broken", "entity"]))[0]; Oz = onmat(z); tz, fz = ty[z], fid[z]
byf = collections.defaultdict(lambda: collections.defaultdict(list))
for j, (f, t) in enumerate(zip(fz, tz)): byf[f][t].append(j)
diffs = [Oz[v["broken"]].mean(0) - Oz[v["entity"]].mean(0) for v in byf.values() if v["broken"] and v["entity"]]
sc = np.mean(diffs, 0); allrate = Oz.mean(0); rb, re_ = Oz[tz == "broken"].mean(0), Oz[tz == "entity"].mean(0)
if args.select == "ratio": sc = np.log((rb + .01) / (re_ + .01)); cap = .2
else: cap = .5
okb, oke = (allrate <= cap) & (rb >= .05), (allrate <= cap) & (re_ >= .05)
def sides(K): return np.argsort(-np.where(okb, sc, -1e9))[:K], np.argsort(-np.where(oke, -sc, -1e9))[:K]
dn20, up20 = sides(20); dn5, _ = sides(5)
print(f"selection: {args.select}; " + f"0/5 rows used: broken {int((tz == 'broken').sum())}, entity {int((tz == 'entity').sum())}; facts with both: {len(diffs)}; best separation broken-side {sc[dn20[0]]:.3f}, entity-side {-sc[up20[0]]:.3f}", flush=True)
# ---- check on facts 1/5..4/5 (not used above)
mi = np.where(mid & np.isin(ty, ["correct", "broken", "entity"]))[0]; Om = onmat(mi); tm, fm = ty[mi], fid[mi]
for K_, (dn, up) in ((20, (dn20, up20)), (5, sides(5))):
    s = Om[:, dn].sum(1).astype(float) - Om[:, up].sum(1); res = []
    for a_, b_ in (("correct", "broken"), ("entity", "broken"), ("correct", "entity")):
        per = [roc_auc_score((tm[m] == b_).astype(int), s[m]) for u in np.unique(fm) for m in [(fm == u) & np.isin(tm, [a_, b_])] if len(set(tm[m])) == 2]
        res.append(f"{a_}-{b_} {np.mean(per):.3f} ({len(per)} facts)")
    print(f"  K={K_} within-fact AUROC on facts 1/5..4/5: " + " | ".join(res), flush=True)
# ---- random latents matched by firing-rate decile
cand = np.where(allrate > .01)[0]; qs = np.quantile(allrate[cand], np.linspace(0, 1, 11)); dec = np.clip(np.digitize(allrate, qs[1:-1]), 0, 9); rr = np.random.default_rng(5); excl = set(dn20) | set(up20)
tgt0 = np.where(mid & (ty == "broken"))[0]; Xt = torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in tgt0]))
def sae_delta(x, lat):                                       # the selected latents' own contribution to the SAE reconstruction of x
    li = torch.as_tensor(lat, device=x.device); pre = ((x - Wg["b_dec"]) if SUB_BDEC else x) @ Wg["W_enc"][:, li] + Wg["b_enc"][li]
    a = pre * (pre > Wg["threshold"][li]); return a @ Wg["W_dec"][li]
def removed(lat):
    if args.remove_mode == "sae": x = Xt.cuda(); return ((sae_delta(x, lat).norm(dim=1) ** 2) / x.norm(dim=1) ** 2).mean().item()
    return ((Xt @ torch.linalg.qr(Wu[lat].T.float())[0]).norm(dim=1) ** 2 / Xt.norm(dim=1) ** 2).mean().item()
def draw(lat, match=False):
    want = removed(lat) if match else None
    for _ in range(2000):
        pl = []
        for j in lat: pl.append(int(rr.choice([c for c in cand if dec[c] == dec[j] and c not in excl and c not in pl])))
        if want is None or abs(removed(pl) - want) <= 0.15 * want: return pl
    raise RuntimeError("no size-matched random set found")
norm = float(np.mean([np.linalg.norm(np.asarray(H[i, 0], dtype=np.float32)) for i in np.random.default_rng(1).choice(len(rows), 500, replace=False)]))
basis = (lambda lat: ("sae", torch.as_tensor(np.array(lat)).cuda())) if args.remove_mode == "sae" else (lambda lat: torch.linalg.qr(Wu[lat].T.float())[0].cuda())
addv = lambda lat, a: (lambda v: (a * norm * v / v.norm()).cuda())(Wu[lat].sum(0).float())
CONDS = [("base", None, args.batch), ("base batch 8 (noise)", None, 8), ("remove BROKEN K20", ("proj", basis(dn20)), args.batch), ("add BROKEN K20 a0.5", ("add", addv(dn20, .5)), args.batch), ("add BROKEN K20 a1", ("add", addv(dn20, 1.)), args.batch),
         ("remove ENTITY-side K20", ("proj", basis(up20)), args.batch), ("add ENTITY-side K20 a0.5", ("add", addv(up20, .5)), args.batch), ("add ENTITY-side K20 a1", ("add", addv(up20, 1.)), args.batch), ("remove BROKEN K5", ("proj", basis(dn5)), args.batch)]
if args.lean: CONDS = [c for c in CONDS if c[0] in ("base", "remove BROKEN K20", "remove BROKEN K5", "remove ENTITY-side K20", "add BROKEN K20 a1", "add ENTITY-side K20 a1")]
print("removed share of ||h||^2 (target rows, step 0): BROKEN K20 %.4f, BROKEN K5 %.4f, ENTITY-side K20 %.4f" % (removed(dn20), removed(dn5), removed(up20)), flush=True)
CONDS += [(f"remove random 20 #{q}", ("proj", basis(draw(dn20, args.match_size))), args.batch) for q in range(args.draws)]
CONDS += [(f"remove random 5 #{q}", ("proj", basis(draw(dn5, args.match_size))), args.batch) for q in range(args.draws)]
CONDS += [(f"add random 20 a1 #{q}", ("add", addv(draw(dn20), 1.)), args.batch) for q in range(args.add_draws)]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id; names = wd._cache.get("names", {}); has_cand = lambda n: (bool(names.get(n)) if n in names else None)
st = {"step": -1, "p": 0, "edit": None}
def hook(module, a, output):
    st["step"] += 1
    if st["edit"] is not None and st["step"] < args.T:
        h = output[0] if isinstance(output, tuple) else output; p = st["p"]; x = h[:, p - 1].float(); kind, v = st["edit"]
        if kind == "proj" and isinstance(v, tuple): y = x - sae_delta(x, v[1])
        elif kind == "proj": y = x - (x @ v) @ v.T
        else: y = x + v[None]
        h[:, p - 1] = y.to(h.dtype)
model.model.layers[LAYER].register_forward_hook(hook)
tgt = np.where(mid & (ty == "broken"))[0]; tgt = tgt[: args.pilot] if args.pilot else tgt
ids = {int(i): tok(build_prompt(rows[i]["question"]))["input_ids"] for i in tgt}; by_len = collections.defaultdict(list)
for i in tgt: by_len[len(ids[int(i)])].append(int(i))
out_f = open(args.out, "w", encoding="utf-8"); R = collections.defaultdict(dict)
for cname, edit, bs in CONDS:
    st["edit"] = edit; tc = time.time()
    for L, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), bs):
            ix = idxs[b0:b0 + bs]; inp = torch.tensor([ids[i] for i in ix], device="cuda"); st["step"] = -1; st["p"] = L
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                             mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, i in enumerate(ix):
                gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]; ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
                ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]}); t3 = "correct" if ok else classify(ans, r["question"], r["relation"], has_cand)[0]
                R[cname][i] = t3; out_f.write(json.dumps({"cond": cname, "row": i, "k": int(k_all[i]), "rule": r["rule"], "answer": ans, "new_type": t3}, ensure_ascii=False) + "\n")
    out_f.flush(); print(f"  {cname}: {time.time() - tc:.0f}s", flush=True)
n = len(tgt); share = lambda c, t: np.mean([R[c][int(i)] == t for i in tgt])
print(f"\n==== broken rows of facts 1/5..4/5 (n={n}) -> new answer type: correct / entity / broken")
for c, *_ in CONDS:
    if "random" not in c: print(f"  {c:<28s} correct {share(c, 'correct'):.3f} | entity {share(c, 'entity'):.3f} | broken {share(c, 'broken'):.3f}")
for pref in ("remove random 20", "remove random 5", "add random 20 a1"):
    cs = [c for c, *_ in CONDS if c.startswith(pref)]
    if cs:
        b = np.array([share(c, "broken") for c in cs]); cc = np.array([share(c, "correct") for c in cs])
        print(f"  {pref:<28s} broken mean {b.mean():.3f} (min {b.min():.3f}, max {b.max():.3f}) | correct mean {cc.mean():.3f} (min {cc.min():.3f}, max {cc.max():.3f})  [{len(cs)} draws]")
sys.stdout.flush(); os._exit(0)
