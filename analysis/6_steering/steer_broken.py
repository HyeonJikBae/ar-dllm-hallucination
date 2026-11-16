"""Does pushing the "correct-like, not broken-like" signal reduce broken answers?   Dream-v0-Base-7B, layer 20, prompt-end token.
Direction: on DEV subjects, rows of facts 1/5..4/5 that are correct vs broken; per latent the t-statistic of the layer-20 SAE activation (prompt end, step 0);
  up-latents = top-K higher in correct rows, down-latents = top-K higher in broken rows;   d = unit( sum unit(Wdec[up]) - sum unit(Wdec[down]) ).
Intervention: at the prompt's last token, layer-20 output  h += alpha * mean||h|| * d  in the first T denoising forwards (T=8; same canvas 32 / 32 steps / topk_margin / temperature 0 as the data).
Conditions: base | base with batch 8 (the same no-intervention run in a different batch layout: bf16 numerics make ~15% of answers differ from the stored ones, so this measures the noise floor) | +d (K=5 / 20, alpha 0.5/1/2) | -d (K=20, alpha 1, the opposite sign) | random direction alpha 1 / 2.
TEST subjects only.  Groups: BROKEN rows of 1/5..4/5 facts (main) | BROKEN rows of 0/5 facts (sample) | CORRECT rows 1/5..4/5 (do we break them?) | ENTITY-wrong rows 1/5..4/5.
Each new answer: correct by the repo's grading rule, otherwise the repo's 3-type rule (classify_rules.classify; names missing from the Wikidata cache stay entity, unverified).
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/6_steering/steer_broken.py [--pilot 10] [--n-b0 150 --n-c 100 --n-e 100] [--batch 16]"""
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
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--out", default="analysis/6_steering/results_broken.jsonl"); ap.add_argument("--T", type=int, default=8)
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
# ---- direction from dev rows (SAE activations of the saved step-0 prompt-end states)
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
dv = np.where(dev & mid & ((ty == "correct") | (ty == "broken")))[0]; yy = (ty[dv] == "correct")
A = torch.cat([enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in dv[b:b + 512]])).cuda()).cpu() for b in range(0, len(dv), 512)]).numpy()
P, Q = A[yy], A[~yy]; tst = (P.mean(0) - Q.mean(0)) / np.sqrt(P.var(0) / len(P) + Q.var(0) / len(Q) + 1e-8)
Wdec = Wd["W_dec"]; Wu = Wdec / Wdec.norm(dim=1, keepdim=True)
def direction(K):
    up, dn = np.argsort(-tst)[:K], np.argsort(tst)[:K]; d = Wu[up].sum(0) - Wu[dn].sum(0); return (d / d.norm()).cuda().to(torch.bfloat16)
print("dev rows for the direction:", len(dv), "(correct", int(yy.sum()), "broken", int((~yy).sum()), ")", flush=True)
norm = float(np.mean([np.linalg.norm(np.asarray(H[i, 0], dtype=np.float32)) for i in r2.choice(len(rows), 500, replace=False)])); print("mean ||h|| =", round(norm, 1))
d5, d20 = direction(5), direction(20); dr = (lambda v: (v / v.norm()).cuda().to(torch.bfloat16))(torch.randn(Wdec.shape[1], generator=torch.Generator().manual_seed(0)))
del enc, Wd; torch.cuda.empty_cache()
T = args.T
CONDS = [("base", None, 0, 0), ("base (batch 8: numerical-noise control)", None, 0, 0), ("+d5 a0.5", d5, .5, T), ("+d5 a1", d5, 1., T), ("+d5 a2", d5, 2., T), ("+d20 a0.5", d20, .5, T), ("+d20 a1", d20, 1., T), ("+d20 a2", d20, 2., T),
         ("-d20 a1 (opposite)", d20, -1., T), ("random a1", dr, 1., T), ("random a2", dr, 2., T)]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id
names = wd._cache.get("names", {}); has_cand = lambda n: (bool(names.get(n)) if n in names else None)
st = {"step": -1, "p": 0, "vec": None, "T": 0}
def hook(module, a, output):
    st["step"] += 1
    if st["vec"] is not None and st["step"] < st["T"]:
        h = output[0] if isinstance(output, tuple) else output; h[:, st["p"] - 1] += st["vec"]
model.model.layers[LAYER].register_forward_hook(hook)
allrows = np.concatenate(list(G.values())); ids = {int(i): tok(build_prompt(rows[i]["question"]))["input_ids"] for i in allrows}
by_len = collections.defaultdict(list)
for i in allrows: by_len[len(ids[int(i)])].append(int(i))
out_f = open(args.out, "w", encoding="utf-8"); res = collections.defaultdict(dict); t0 = time.time()
for name, d, a, Tn in CONDS:
    st["vec"] = None if d is None else (a * norm * d); st["T"] = Tn; tc = time.time()
    for L, idxs in sorted(by_len.items()):
        bs = 8 if name.startswith("base (batch 8") else args.batch
        for b0 in range(0, len(idxs), bs):
            ix = idxs[b0:b0 + bs]; inp = torch.tensor([ids[i] for i in ix], device="cuda"); st["step"] = -1; st["p"] = L
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin",
                                             alg_temp=0.0, mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, i in enumerate(ix):
                gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]
                ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
                ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]})
                t3 = "correct" if ok else classify(ans, r["question"], r["relation"], has_cand)[0]
                res[name][i] = (ans, t3)
                out_f.write(json.dumps({"cond": name, "row": i, "sample_id": r["sample_id"], "version": r["version"], "k": int(k[i]), "old_type": ty[i], "answer": ans, "new_type": t3, "old_answer": r["model_answer"]}, ensure_ascii=False) + "\n")
    out_f.flush(); print(f"[{name}] done in {time.time() - tc:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
print("\nnew answer types by group and condition: correct / entity / broken / abstain (share of the group); 'same' = answer identical to the base run")
for gname, gi in G.items():
    print(f"\n== {gname} (n={len(gi)})")
    for name, *_ in CONDS:
        c = collections.Counter(res[name][int(i)][1] for i in gi); n = len(gi); same = np.mean([res[name][int(i)][0] == res["base"][int(i)][0] for i in gi])
        print(f"  {name:<20s} correct {c['correct']/n:.3f} | entity {c['entity']/n:.3f} | broken {c['broken']/n:.3f} | abstain {c['abstain']/n:.3f} | same as base {same:.2f}")
sys.stdout.flush(); os._exit(0)
