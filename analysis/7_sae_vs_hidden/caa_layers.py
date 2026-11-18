"""Does the steering effect depend on the layer?  Direction B added at layer 16, 20 or 24.   Dream-v0-Base-7B, prompt-end token, all 32 forwards.
B_L = unit( mean over facts of [mean(correct h_L) - mean(wrong h_L)] ), facts 1/5..4/5, from the other 4 subject folds (same folds as analysis/6_steering),
h_L = step-0 prompt-end state at layer L (data/1_raw/dream_hidden_layers, extract_layers.py).  Correct vs wrong only (wrong types merged).
Added as  c * mean||h_L|| * B_L  (c in {0.5, 1}), so the push has the same size RELATIVE to the state at each layer.
Control per layer: D_L = wrong rows of facts 1/5..4/5 (other folds) split at random into two halves, mean difference (a meaningless direction), made orthogonal to B_L, c = 1.
(The raw split-half difference can be strongly (anti-)aligned with B because both pick up the high-variance axes of the state.)
Targets (facts 1/5..4/5, the same rows for every condition): 300 random wrong rows, 300 random correct rows.
Usage (repo root): CUDA_VISIBLE_DEVICES=2 python3 analysis/7_sae_vs_hidden/caa_layers.py"""
import collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading")
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
D_ = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; CANVAS, STEPS, BS, NF = 32, 32, 16, 5
LAYERS = (16, 20, 24)
rows = [json.loads(l) for l in open(f"{D_}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D_}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); order = {x: j for j, x in enumerate(subs)}; grp = np.array([order[x] % NF for x in sub])
k = np.array([r["k_correct"] for r in rows]); cor = np.array([r["correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
H = np.load(f"{D_}/1_raw/dream_hidden_layers/hid_prompt_last_layers.npy", mmap_mode="r")
unit = lambda v: v / np.linalg.norm(v)
dirs, norm = {}, {}
for L in LAYERS:
    X = np.asarray(H[:, L], dtype=np.float32); norm[L] = float(np.linalg.norm(X[np.random.default_rng(1).choice(len(X), 500, replace=False)], axis=1).mean())
    for g in range(NF):
        tr = grp != g; byf = collections.defaultdict(lambda: [[], []])
        for i in np.where(tr & mid)[0]: byf[fid[i]][int(cor[i])].append(i)
        B = unit(np.mean([X[c].mean(0) - X[w].mean(0) for w, c in byf.values() if w and c], 0))
        wi = np.random.default_rng(100 + g).permutation(np.where(tr & mid & ~cor)[0])
        Dr = unit(X[wi[: len(wi) // 2]].mean(0) - X[wi[len(wi) // 2:]].mean(0)); cBD = float(Dr @ B)
        dirs[(L, g)] = {"B": B, "D": unit(Dr - (Dr @ B) * B), "cos_raw": cBD}     # D made orthogonal to B, so the control does not push along (or against) B
    print(f"layer {L}: mean||h|| {norm[L]:.1f} | cos(B, raw D) per fold {[round(dirs[(L, g)]['cos_raw'], 2) for g in range(NF)]} (D is then made orthogonal to B)", flush=True)
r5 = np.random.default_rng(5)
G = {"wrong": r5.permutation(np.where(mid & ~cor)[0])[:300], "correct": r5.permutation(np.where(mid & cor)[0])[:300]}
tgt = np.concatenate(list(G.values()))
CONDS = [("base", None, None, 0.)] + [(f"L{L} B c{c}", L, "B", c) for L in LAYERS for c in (.5, 1.)] + [(f"L{L} D c1", L, "D", 1.) for L in LAYERS]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id
st = {"p": 0, "L": None, "vec": None}
def hook(L):
    def f(module, a, output):
        if st["L"] == L:
            h = output[0] if isinstance(output, tuple) else output; p = st["p"]; h[:, p - 1] = (h[:, p - 1].float() + st["vec"]).to(h.dtype)
    return f
for L in LAYERS: model.model.layers[L].register_forward_hook(hook(L))
ids = {int(i): tok(build_prompt(rows[i]["question"]))["input_ids"] for i in tgt}; by_len = collections.defaultdict(list)
for i in tgt: by_len[len(ids[int(i)])].append(int(i))
R = collections.defaultdict(dict); out = open("analysis/7_sae_vs_hidden/results_caa_layers.jsonl", "w", encoding="utf-8")
for cname, L, d, c in CONDS:
    tc = time.time()
    for pl, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), BS):
            ix = idxs[b0:b0 + BS]; inp = torch.tensor([ids[i] for i in ix], device="cuda"); st["p"] = pl; st["L"] = L
            st["vec"] = None if d is None else torch.from_numpy(np.stack([dirs[(L, grp[i])][d] * c * norm[L] for i in ix])).float().cuda()
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                             mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, i in enumerate(ix):
                gen = o[j][pl:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]; ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
                ok = bool(is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]}))
                R[cname][i] = ok; out.write(json.dumps({"cond": cname, "row": i, "fold": int(grp[i]), "k": int(k[i]), "was_correct": bool(cor[i]), "answer": ans, "correct": ok}, ensure_ascii=False) + "\n")
    out.flush(); print(f"  {cname}: {time.time() - tc:.0f}s", flush=True)
acc = lambda c, g, sel=None: np.mean([R[c][int(i)] for i in (G[g] if sel is None else G[g][sel])])
print("\nshare correct after the edit.   wrong rows: higher = more fixed;  correct rows: higher = fewer broken")
print(f"  {'condition':<14s} wrong (n={len(G['wrong'])})  correct (n={len(G['correct'])})  | wrong rows by k: 1/5 2/5 3/5 4/5")
for cname, *_ in CONDS:
    kw = k[G["wrong"]]
    print(f"  {cname:<14s} {acc(cname, 'wrong'):.3f}          {acc(cname, 'correct'):.3f}          | " + " ".join(f"{acc(cname, 'wrong', kw == q):.2f}" for q in range(1, 5)))
print("  wrong rows by k:", {q: int((k[G['wrong']] == q).sum()) for q in range(1, 5)})
sys.stdout.flush(); os._exit(0)
