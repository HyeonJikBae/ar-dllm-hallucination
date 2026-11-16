"""Cross-fitted test: find the CORRECT signal and the BROKEN signal inside the same facts, then steer the broken signal and see how many broken answers are resolved.   Dream-v0-Base-7B, layer 20, prompt-end token, SAE (16k).
Folds: subjects split in two halves (the repo's dev / test halves).  Signal found on one half, steering evaluated on the BROKEN rows (facts 1/5..4/5) of the other half; then the roles swap (about 310 broken rows in all).
Signal (per source half, step-0 SAE activations at the prompt-end token, rows of facts 1/5..4/5):
  per fact with both kinds of rows: (firing rate in correct rows) - (firing rate in broken rows), averaged over the facts of each k group (k = 4, 3, 2, 1);  same for entity-wrong vs broken and correct vs entity (pooled over k).
  CORRECT signal latent score = min over groups {correct-broken for k=1..4, correct-entity};  BROKEN signal latent score = min over groups {broken-correct for k=1..4, broken-entity}  (MaxMin, like the paper over entity types).
  Latents firing on > 50% of all rows are dropped, and a signal latent must fire in >= 5% of the rows of its own kind.   Top K=20 each.
Edits at the prompt-end token's layer-20 output in the first T=8 denoising forwards:  remove = project out the span of the 20 decoder directions;  add = add alpha*mean||h|| * unit(sum of the 20 unit decoder directions).
Conditions: base | base batch 8 (noise) | remove broken-signal | add broken-signal a0.5, a1 (should INCREASE broken) | remove correct-signal | add correct-signal a0.5, a1 | R random-latent controls (remove 20 random latents matched on firing-rate decile).
New answers are graded / classified with the repo rules.   Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/6_steering/crossfit_broken.py [--pilot 8] [--draws 8]"""
import argparse, collections, json, os, sys, time
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "analysis"); sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import sae_io, wd
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
from classify_rules import classify
ap = argparse.ArgumentParser()
ap.add_argument("--pilot", type=int, default=0); ap.add_argument("--draws", type=int, default=8); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--T", type=int, default=8); ap.add_argument("--K", type=int, default=20)
ap.add_argument("--nfold", type=int, default=0, help="subject-level n-fold cross-fitting (signal from the other n-1 folds, steering on the held-out fold)"); ap.add_argument("--all", action="store_true", help="no split: signal found on ALL rows of facts 1/5..4/5, steering on all their broken rows (in-sample, an upper bound)"); ap.add_argument("--out", default="analysis/6_steering/results_crossfit.jsonl")
args = ap.parse_args()
D = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS = 20, 32, 32; K = args.K
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); half = np.isin(sub, subs[: len(subs) // 2])
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k_all = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k_all >= 1) & (k_all <= 4)
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
Wu = Wd["W_dec"] / Wd["W_dec"].norm(dim=1, keepdim=True); H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
def encode(idx):
    return torch.cat([enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in idx[b:b + 512]])).cuda()).cpu() for b in range(0, len(idx), 512)]).numpy()
def find_signal(src):
    idx = np.where(src & mid & np.isin(ty, ["correct", "broken", "entity"]))[0]; A = encode(idx); on = (A > 0).astype(np.float32); t = ty[idx]; f = fid[idx]; kk = k_all[idx]
    byf = collections.defaultdict(lambda: {"idx": collections.defaultdict(list), "k": 0})
    for j, (ff, tt, k_) in enumerate(zip(f, t, kk)): byf[ff]["idx"][tt].append(j); byf[ff]["k"] = k_
    rate = lambda f_, tt: on[byf[f_]["idx"][tt]].mean(0)
    d = {g: [] for g in (1, 2, 3, 4)}; e, ce = [], []
    for ff, v in byf.items():
        if v["idx"]["correct"] and v["idx"]["broken"]: d[v["k"]].append(rate(ff, "correct") - rate(ff, "broken"))
        if v["idx"]["entity"] and v["idx"]["broken"]: e.append(rate(ff, "entity") - rate(ff, "broken"))
        if v["idx"]["correct"] and v["idx"]["entity"]: ce.append(rate(ff, "correct") - rate(ff, "entity"))
    groups_c = [np.mean(d[g], 0) for g in d if d[g]] + [np.mean(ce, 0)]; groups_b = [-np.mean(d[g], 0) for g in d if d[g]] + [-np.mean(e, 0)]
    sc_c, sc_b = np.min(groups_c, 0), np.min(groups_b, 0); allrate = on.mean(0)
    okc = (allrate <= 0.5) & (on[t == "correct"].mean(0) >= 0.05); okb = (allrate <= 0.5) & (on[t == "broken"].mean(0) >= 0.05)
    up = np.argsort(-np.where(okc, sc_c, -9))[:K]; dn = np.argsort(-np.where(okb, sc_b, -9))[:K]
    info = dict(groups={g: len(d[g]) for g in d}, ent_broken=len(e), cor_ent=len(ce), score_c=float(sc_c[up[0]]), score_b=float(sc_b[dn[0]]))
    return up, dn, allrate, idx, A, info
norm = float(np.mean([np.linalg.norm(np.asarray(H[i, 0], dtype=np.float32)) for i in np.random.default_rng(1).choice(len(rows), 500, replace=False)]))
folds = []
_order = {x: j for j, x in enumerate(subs)}; _grp = np.array([_order[x] % max(args.nfold, 1) for x in sub]) if args.nfold else None
FOLDDEF = [(f"F{g}: signal from the other {args.nfold - 1} folds, steer fold {g}", _grp != g, _grp == g) for g in range(args.nfold)] if args.nfold else [("ALL: signal from all 1/5-4/5 rows, steer all their broken rows", np.ones(len(rows), bool), np.ones(len(rows), bool))] if args.all else [("A: signal from half 1, steer half 2", half, ~half), ("B: signal from half 2, steer half 1", ~half, half)]
for name, src_half, tgt_half in FOLDDEF:
    up, dn, allrate, idx, A, info = find_signal(src_half); folds.append((name, up, dn, allrate, tgt_half)); print(f"\n[{name}] facts per group {info['groups']}, entity-broken facts {info['ent_broken']}, correct-entity facts {info['cor_ent']}; best MaxMin score correct {info['score_c']:.3f}, broken {info['score_b']:.3f}", flush=True)
    # validation on the held-out half: within-fact AUROC of (#broken-signal latents ON - #correct-signal latents ON)
    vi = np.where(tgt_half & mid & np.isin(ty, ["correct", "broken", "entity"]))[0]; Av = encode(vi); s = (Av[:, dn] > 0).sum(1) - (Av[:, up] > 0).sum(1); tv, fv = ty[vi], fid[vi]
    for a_, b_ in (("correct", "broken"), ("entity", "broken"), ("correct", "entity")):
        per = []
        for u in np.unique(fv):
            m = (fv == u) & np.isin(tv, [a_, b_])
            if len({x for x in tv[m]}) == 2: per.append(roc_auc_score((tv[m] == b_).astype(int), s[m]) if b_ != "entity" else roc_auc_score((tv[m] == "entity").astype(int), s[m]))
        print(f"   held-out within-fact AUROC, score vs {a_}-{b_} ({len(per)} facts): {np.mean(per):.3f}")
if args.pilot: pass
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id; names = wd._cache.get("names", {}); has_cand = lambda n: (bool(names.get(n)) if n in names else None)
st = {"step": -1, "p": 0, "edit": None}
def hook(module, a, output):
    st["step"] += 1
    if st["edit"] is not None and st["step"] < args.T:
        h = output[0] if isinstance(output, tuple) else output; p = st["p"]; x = h[:, p - 1].float(); kind, v = st["edit"]
        y = x - (x @ v) @ v.T if kind == "proj" else x + v[None]
        h[:, p - 1] = y.to(h.dtype)
model.model.layers[LAYER].register_forward_hook(hook)
basis = lambda lat: torch.linalg.qr(Wu[lat].T.float())[0].cuda()
addv = lambda lat, a: (lambda v: (a * norm * v / v.norm()).cuda())(Wu[lat].sum(0).float())
out_f = open(args.out, "w", encoding="utf-8"); results = collections.defaultdict(dict)
for fname, up, dn, allrate, tgt in folds:
    tgt_rows = np.where(tgt & mid & (ty == "broken"))[0]; tgt_rows = tgt_rows[: args.pilot] if args.pilot else tgt_rows
    ids = {int(i): tok(build_prompt(rows[i]["question"]))["input_ids"] for i in tgt_rows}; by_len = collections.defaultdict(list)
    for i in tgt_rows: by_len[len(ids[int(i)])].append(int(i))
    # random-latent controls: 20 latents per draw, each matched to a signal latent by firing-rate decile
    cand = np.where(allrate > 0.01)[0]; qs = np.quantile(allrate[cand], np.linspace(0, 1, 11)); dec = np.clip(np.digitize(allrate, qs[1:-1]), 0, 9)
    rr = np.random.default_rng(5); draws = []
    for _ in range(args.draws):
        pickl = []
        for j in dn:
            pool = [c for c in cand if dec[c] == dec[j] and c not in set(dn) and c not in set(up) and c not in pickl]; pickl.append(int(rr.choice(pool)))
        draws.append(pickl)
    CONDS = [("base", None, args.batch), ("base batch 8 (noise)", None, 8), ("remove BROKEN signal", ("proj", basis(dn)), args.batch), ("add BROKEN signal a0.5", ("add", addv(dn, .5)), args.batch), ("add BROKEN signal a1", ("add", addv(dn, 1.)), args.batch),
             ("remove CORRECT signal", ("proj", basis(up)), args.batch), ("add CORRECT signal a0.5", ("add", addv(up, .5)), args.batch), ("add CORRECT signal a1", ("add", addv(up, 1.)), args.batch)] + \
            [(f"remove random latents #{q}", ("proj", basis(lat)), args.batch) for q, lat in enumerate(draws)]
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
                    results[cname][i] = t3; out_f.write(json.dumps({"fold": fname[0], "cond": cname, "row": i, "sample_id": r["sample_id"], "k": int(k_all[i]), "rule": r["rule"], "answer": ans, "new_type": t3}, ensure_ascii=False) + "\n")
        out_f.flush(); print(f"  [{fname[0]}] {cname}: {time.time() - tc:.0f}s", flush=True)
print("\n==== pooled over both folds: the BROKEN rows (facts 1/5..4/5) -> new answer type")
names_c = list(results.keys()); n = len(results["base"])
for c in names_c:
    cnt = collections.Counter(results[c].values()); print(f"  {c:<32s} correct {cnt['correct']/n:.3f} | entity {cnt['entity']/n:.3f} | broken {cnt['broken']/n:.3f} | abstain {cnt['abstain']/n:.3f}")
rb = np.array([np.mean([t == "broken" for t in results[c].values()]) for c in names_c if c.startswith("remove random")])
print(f"\nrandom-latent controls: broken rate mean {rb.mean():.3f}, min {rb.min():.3f}, max {rb.max():.3f} (n draws per fold {args.draws}, pooled over folds)")
print("n broken rows:", n); sys.stdout.flush(); os._exit(0)
