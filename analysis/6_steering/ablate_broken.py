"""Remove the "broken" signal instead of adding the "correct" one, optionally only when a gate says the row looks broken.   Dream-v0-Base-7B, layer 20, prompt-end token.
Signal: on DEV subjects, correct vs broken rows of 1/5..4/5 facts, per latent the t-statistic of the layer-20 SAE activation (prompt end, step 0).  K=20 latents per side.
  broken-side latents (dn20) = higher in broken rows;  correct-side latents (up20) = higher in correct rows.
  d  = unit( sum unit(Wdec[dn20]) - sum unit(Wdec[up20]) )   (points toward "broken")        Q = orthonormal basis of the 20 broken-side decoder directions
Gate: score = (#broken-side latents ON) - (#correct-side latents ON) at the prompt-end token of the first forward (step 0, before any change);  fires when score > the 90th percentile of the DEV correct rows (about 10% false alarms).
Edits at the prompt-end token's layer-20 output, in the first T=8 denoising forwards, on every row (always) or on gated rows only (gated):
  proj d      h <- h - (h.d) d                     remove the component along d
  meanset d   h <- h - (h.d - c) d                 set that component to c = mean (h.d) of DEV correct rows (remove only the excess)
  proj Q      h <- h - Q Q^T h                     remove the 20-dim subspace
Controls: the same removal along a random unit direction / a random 20-dim subspace.   Base = no edit, same batch layout.
TEST subjects only, same four row groups as steer_broken.py.  New answers are graded / classified with the repo rules.
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/6_steering/ablate_broken.py [--pilot 8]"""
import argparse, collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "analysis"); sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import sae_io, wd
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
from classify_rules import classify
ap = argparse.ArgumentParser()
ap.add_argument("--pilot", type=int, default=0); ap.add_argument("--n-b0", type=int, default=150); ap.add_argument("--n-c", type=int, default=100); ap.add_argument("--n-e", type=int, default=100)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", default="analysis/6_steering/results_ablate.jsonl"); ap.add_argument("--T", type=int, default=8); ap.add_argument("--K", type=int, default=20)
args = ap.parse_args()
D = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS = 20, 32, 32
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); dev = np.isin(sub, subs[: len(subs) // 2]); te = ~dev
cor = np.array([r["correct"] for r in rows]); k = np.array([r["k_correct"] for r in rows]); ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); mid = (k >= 1) & (k <= 4)
r2 = np.random.default_rng(1); pick = lambda m, n: r2.permutation(np.where(m)[0])[:n]
G = {"broken 1-4/5 (main)": np.where(te & mid & (ty == "broken"))[0], "broken 0/5": pick(te & (k == 0) & (ty == "broken"), args.n_b0),
     "correct 1-4/5": pick(te & mid & cor, args.n_c), "entity 1-4/5": pick(te & mid & (ty == "entity"), args.n_e)}
if args.pilot: G = {n: v[: args.pilot] for n, v in G.items()}
print({n: len(v) for n, v in G.items()}, flush=True)
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
dv = np.where(dev & mid & ((ty == "correct") | (ty == "broken")))[0]; yy = (ty[dv] == "correct")
R = np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in dv])
A = torch.cat([enc(torch.from_numpy(R[b:b + 512]).cuda()).cpu() for b in range(0, len(R), 512)]).numpy()
P, Q_ = A[yy], A[~yy]; tst = (P.mean(0) - Q_.mean(0)) / np.sqrt(P.var(0) / len(P) + Q_.var(0) / len(Q_) + 1e-8)
K = args.K; up, dn = np.argsort(-tst)[:K], np.argsort(tst)[:K]
Wdec = Wd["W_dec"]; Wu = Wdec / Wdec.norm(dim=1, keepdim=True)
d = (Wu[dn].sum(0) - Wu[up].sum(0)); d = (d / d.norm()).float()
Qb, _ = torch.linalg.qr(Wu[dn].T.float())                                       # [3584, K]
g_ = torch.Generator().manual_seed(0); dr = torch.randn(Wdec.shape[1], generator=g_); dr = dr / dr.norm(); Qr, _ = torch.linalg.qr(torch.randn(Wdec.shape[1], K, generator=g_))
c_mean = float((torch.from_numpy(R[yy]) @ d).mean())
s_dev = (A[:, dn] > 0).sum(1) - (A[:, up] > 0).sum(1); thr = float(np.percentile(s_dev[yy], 90))
print(f"dev rows {len(dv)} (correct {int(yy.sum())}, broken {int((~yy).sum())}); gate threshold (score > {thr:.0f}): dev false-alarm rate {(s_dev[yy] > thr).mean():.3f}, dev detection rate of broken {(s_dev[~yy] > thr).mean():.3f}", flush=True)
print("mean component along d: correct rows %.2f, broken rows %.2f" % (c_mean, float((torch.from_numpy(R[~yy]) @ d).mean())), flush=True)
dn_t, up_t = torch.tensor(dn).cuda(), torch.tensor(up).cuda()
d, Qb, dr, Qr = d.cuda(), Qb.cuda(), dr.cuda(), Qr.cuda()
CONDS = [("base", None, False), ("proj d, always", ("proj", d), False), ("proj d, gated", ("proj", d), True), ("meanset d, always", ("mean", d), False), ("meanset d, gated", ("mean", d), True),
         ("proj Q20, always", ("projQ", Qb), False), ("proj Q20, gated", ("projQ", Qb), True), ("random proj d, always", ("proj", dr), False), ("random proj Q20, always", ("projQ", Qr), False), ("random proj d, gated", ("proj", dr), True)]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id
names = wd._cache.get("names", {}); has_cand = lambda n: (bool(names.get(n)) if n in names else None)
st = {"step": -1, "p": 0, "edit": None, "gated": False, "gate": None, "T": args.T}
def hook(module, a, output):
    st["step"] += 1
    h = output[0] if isinstance(output, tuple) else output; p = st["p"]
    if st["step"] == 0:                                                           # gate decision on the unedited first-forward state
        on = (enc(h[:, p - 1].float()) > 0); st["gate"] = (on[:, dn_t].sum(1) - on[:, up_t].sum(1)) > thr
    if st["edit"] is not None and st["step"] < st["T"]:
        kind, v = st["edit"]; x = h[:, p - 1].float()
        if kind == "proj": y = x - (x @ v)[:, None] * v[None]
        elif kind == "mean": y = x - ((x @ v) - c_mean)[:, None] * v[None]
        else: y = x - (x @ v) @ v.T
        if st["gated"]: y = torch.where(st["gate"][:, None], y, x)
        h[:, p - 1] = y.to(h.dtype)
model.model.layers[LAYER].register_forward_hook(hook)
allrows = np.concatenate(list(G.values())); ids = {int(i): tok(build_prompt(rows[i]["question"]))["input_ids"] for i in allrows}
by_len = collections.defaultdict(list)
for i in allrows: by_len[len(ids[int(i)])].append(int(i))
out_f = open(args.out, "w", encoding="utf-8"); res = collections.defaultdict(dict); gate_of = {}; t0 = time.time()
for name, edit, gated in CONDS:
    st["edit"] = edit; st["gated"] = gated; tc = time.time()
    for L, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), args.batch):
            ix = idxs[b0:b0 + args.batch]; inp = torch.tensor([ids[i] for i in ix], device="cuda"); st["step"] = -1; st["p"] = L
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin",
                                             alg_temp=0.0, mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gt = st["gate"].cpu().tolist()
            for j, i in enumerate(ix):
                gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]
                ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
                ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]})
                t3 = "correct" if ok else classify(ans, r["question"], r["relation"], has_cand)[0]
                res[name][i] = (ans, t3); gate_of[i] = bool(gt[j])
                out_f.write(json.dumps({"cond": name, "row": i, "sample_id": r["sample_id"], "version": r["version"], "k": int(k[i]), "old_type": ty[i], "gate": bool(gt[j]), "answer": ans, "new_type": t3, "old_answer": r["model_answer"]}, ensure_ascii=False) + "\n")
    out_f.flush(); print(f"[{name}] done in {time.time() - tc:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
print("\ngate fire rate by group (test rows):", {gn: round(float(np.mean([gate_of[int(i)] for i in gi])), 3) for gn, gi in G.items()})
print("\nnew answer types by group and condition: correct / entity / broken / abstain (share of the group); 'same' = answer identical to the base run")
for gname, gi in G.items():
    print(f"\n== {gname} (n={len(gi)})")
    for name, *_ in CONDS:
        c = collections.Counter(res[name][int(i)][1] for i in gi); n = len(gi); same = np.mean([res[name][int(i)][0] == res["base"][int(i)][0] for i in gi])
        print(f"  {name:<26s} correct {c['correct']/n:.3f} | entity {c['entity']/n:.3f} | broken {c['broken']/n:.3f} | abstain {c['abstain']/n:.3f} | same as base {same:.2f}")
sys.stdout.flush(); os._exit(0)
