"""Signal-based steering with a mean-difference direction (CAA, contrastive activation addition), cross-fitted.   Dream-v0-Base-7B, layer 20, prompt-end token, all 32 forwards.
Targets: the 310 broken rows of facts 1/5..4/5.   5 subject folds (as crossfit_broken.py --nfold 5): a target's direction is computed only from the other 4 folds' rows.
Directions (from saved step-0 prompt-end states, other folds only):
  A  correct - broken, pooled over all rows (all k)                    -> "answering state" minus "broken state"
  B  correct - broken, within the same fact (facts 1/5..4/5), averaged  -> fact difficulty removed
  C  entity  - broken, pooled                                            (control: another wrong-answer state)
  D  broken(half 1) - broken(half 2), pooled                             (control: a direction inside the broken rows' own spread)
  R  random directions: unit sum of the decoder vectors of 20 random SAE latents that fire at this position (on-manifold)
Every direction is scaled to the same length  c * mean||h||  (mean||h|| ~ 54), c in {0.25, 0.5, 1}, and ADDED to the prompt-end state.
Reference: the patch experiment (patch_upper.py) replaced the state; its mean distance ||h_donor - h|| is printed.
Usage: CUDA_VISIBLE_DEVICES=2 python3 analysis/6_steering/caa_broken.py"""
import collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "analysis"); sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import sae_io, wd
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
from classify_rules import classify
D = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS, BS, NF = 20, 32, 32, 16, 5
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); order = {x: j for j, x in enumerate(subs)}; grp = np.array([order[x] % NF for x in sub])
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r"); X = np.asarray(H[:, 0], dtype=np.float32); norm = float(np.linalg.norm(X[np.random.default_rng(1).choice(len(X), 500, replace=False)], axis=1).mean())
tgt = np.where(mid & (ty == "broken"))[0]
unit = lambda v: v / np.linalg.norm(v)
dirs = {}
for g in range(NF):
    tr = grp != g; c_, b_, e_ = tr & (ty == "correct"), tr & (ty == "broken"), tr & (ty == "entity")
    bi = np.where(b_)[0]; r2 = np.random.default_rng(10 + g).permutation(bi); h1, h2 = r2[: len(bi) // 2], r2[len(bi) // 2:]
    within = []; byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & np.isin(ty, ["correct", "broken"]))[0]: byf[fid[i]][ty[i]].append(i)
    for v in byf.values():
        if v["correct"] and v["broken"]: within.append(X[v["correct"]].mean(0) - X[v["broken"]].mean(0))
    dirs[g] = {"A": unit(X[c_].mean(0) - X[b_].mean(0)), "B": unit(np.mean(within, 0)), "C": unit(X[e_].mean(0) - X[b_].mean(0)), "D": unit(X[h1].mean(0) - X[h2].mean(0))}
    if g == 0: print(f"fold 0: raw lengths  A {np.linalg.norm(X[c_].mean(0) - X[b_].mean(0)):.2f}  B {np.linalg.norm(np.mean(within, 0)):.2f} ({len(within)} facts)  C {np.linalg.norm(X[e_].mean(0) - X[b_].mean(0)):.2f}  D {np.linalg.norm(X[h1].mean(0) - X[h2].mean(0)):.2f}; mean||h|| {norm:.1f}")
cos = lambda a, b: float(a @ b)
print("cosine between directions (fold 0): A-B %.2f  A-C %.2f  B-C %.2f  A-D %.2f" % (cos(dirs[0]["A"], dirs[0]["B"]), cos(dirs[0]["A"], dirs[0]["C"]), cos(dirs[0]["B"], dirs[0]["C"]), cos(dirs[0]["A"], dirs[0]["D"])))
print("fold overlap: cosine of A between folds 0 and 1: %.3f, of B: %.3f" % (cos(dirs[0]["A"], dirs[1]["A"]), cos(dirs[0]["B"], dirs[1]["B"])))
# patch reference distance
pj = [json.loads(l) for l in open("analysis/6_steering/results_patch.jsonl")] if os.path.exists("analysis/6_steering/results_patch.jsonl") else []
pd_ = [np.linalg.norm(X[r["donor"]] - X[r["row"]]) for r in pj if r["cond"].startswith("OTHER fact CORRECT") and r["donor"] is not None]
if pd_: print(f"patch reference: mean ||h_donor - h|| = {np.mean(pd_):.1f} ({np.mean(pd_) / norm:.2f} x mean||h||)")
# random on-manifold directions
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, True)
fire = np.zeros(16384)
for b in range(0, len(X), 4096): fire += (enc(torch.from_numpy(X[b:b + 4096]).cuda()) > 0).float().sum(0).cpu().numpy()
cand = np.where(fire / len(X) >= .01)[0]; Wdec = Wd["W_dec"].numpy(); Wu = Wdec / np.linalg.norm(Wdec, axis=1, keepdims=True); rr = np.random.default_rng(7)
RAND = [unit(Wu[rr.choice(cand, 20, replace=False)].sum(0)) for _ in range(6)]
del enc; torch.cuda.empty_cache()
CONDS = [("base", None, 0)] + [(f"A correct-broken pooled c{c}", "A", c) for c in (.25, .5, 1)] + [(f"B correct-broken within-fact c{c}", "B", c) for c in (.25, .5, 1)] + \
        [(f"C entity-broken c{c}", "C", c) for c in (.5, 1)] + [(f"D broken-broken c{c}", "D", c) for c in (.5, 1)] + [(f"R random #{q} c{c}", q, c) for c in (.5, 1) for q in range(len(RAND))]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id; names = wd._cache.get("names", {}); has_cand = lambda n: (bool(names.get(n)) if n in names else None)
st = {"step": -1, "p": 0, "vec": None}
def hook(module, a, output):
    st["step"] += 1
    if st["vec"] is not None:
        h = output[0] if isinstance(output, tuple) else output; p = st["p"]; h[:, p - 1] = (h[:, p - 1].float() + st["vec"]).to(h.dtype)
model.model.layers[LAYER].register_forward_hook(hook)
ids = {int(i): tok(build_prompt(rows[i]["question"]))["input_ids"] for i in tgt}; by_len = collections.defaultdict(list)
for i in tgt: by_len[len(ids[int(i)])].append(int(i))
R = collections.defaultdict(dict); out = open("analysis/6_steering/results_caa.jsonl", "w", encoding="utf-8")
for cname, which, c in CONDS:
    tc = time.time()
    for L, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), BS):
            ix = idxs[b0:b0 + BS]; inp = torch.tensor([ids[i] for i in ix], device="cuda"); st["step"] = -1; st["p"] = L
            if which is None: st["vec"] = None
            else: st["vec"] = torch.from_numpy(np.stack([(RAND[which] if isinstance(which, int) else dirs[grp[i]][which]) * c * norm for i in ix])).float().cuda()
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                             mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, i in enumerate(ix):
                gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]; ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
                ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]}); t3 = "correct" if ok else classify(ans, r["question"], r["relation"], has_cand)[0]
                R[cname][i] = t3; out.write(json.dumps({"cond": cname, "row": i, "fold": int(grp[i]), "k": int(k[i]), "rule": r["rule"], "answer": ans, "new_type": t3}, ensure_ascii=False) + "\n")
    out.flush(); print(f"  {cname}: {time.time() - tc:.0f}s", flush=True)
n = len(tgt); sh = lambda c, t: np.mean([R[c][int(i)] == t for i in tgt])
print(f"\n==== broken rows of facts 1/5..4/5 (n={n}) -> new answer type")
for cname, which, c in CONDS:
    if isinstance(which, int): continue
    print(f"  {cname:<36s} correct {sh(cname, 'correct'):.3f} | entity {sh(cname, 'entity'):.3f} | broken {sh(cname, 'broken'):.3f}")
for c in (.5, 1):
    cs = [cn for cn, w, cc in CONDS if isinstance(w, int) and cc == c]; co = np.array([sh(x, "correct") for x in cs]); br = np.array([sh(x, "broken") for x in cs])
    print(f"  random (on-manifold) c{c}: correct mean {co.mean():.3f} ({co.min():.3f}-{co.max():.3f}) | broken mean {br.mean():.3f} ({br.min():.3f}-{br.max():.3f})  [{len(cs)} draws]")
sys.stdout.flush(); os._exit(0)
