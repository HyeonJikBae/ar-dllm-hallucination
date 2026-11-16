"""Outcomes of direction B for (almost) every kind of row, so that gating rules can be simulated offline.
Rows: ALL rows of facts 1/5..4/5 (5,950) + 1,000 random rows of 0/5 facts + 1,000 random rows of 5/5 facts.
Conditions: base | B c0.5 | B c1  (B = within-fact correct-minus-broken mean difference, from the other 4 subject folds; added at the prompt-end token, layer 20, all 32 forwards; c * mean||h||).
A row's edit never depends on the other rows of its batch, so "steer only the rows a gate selects" = take the steered outcome for selected rows and the base outcome for the rest.
Usage: CUDA_VISIBLE_DEVICES=2 python3 analysis/6_steering/caa_all_rows.py"""
import collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import wd
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
from classify_rules import classify
D = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS, BS, NF = 20, 32, 32, 16, 5
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); order = {x: j for j, x in enumerate(subs)}; grp = np.array([order[x] % NF for x in sub])
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r"); X = np.asarray(H[:, 0], dtype=np.float32); norm = float(np.linalg.norm(X[np.random.default_rng(1).choice(len(X), 500, replace=False)], axis=1).mean())
unit = lambda v: v / np.linalg.norm(v); B = {}
for g in range(NF):
    byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where((grp != g) & mid & np.isin(ty, ["correct", "broken"]))[0]: byf[fid[i]][ty[i]].append(i)
    B[g] = unit(np.mean([X[v["correct"]].mean(0) - X[v["broken"]].mean(0) for v in byf.values() if v["correct"] and v["broken"]], 0))
np.save("analysis/6_steering/B_directions.npy", np.stack([B[g] for g in range(NF)]))
r4 = np.random.default_rng(4)
tgt = np.concatenate([np.where(mid)[0], r4.permutation(np.where(k == 0)[0])[:1000], r4.permutation(np.where(k == 5)[0])[:1000]])
print("rows", len(tgt), flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id; names = wd._cache.get("names", {}); has_cand = lambda n: (bool(names.get(n)) if n in names else None)
st = {"p": 0, "vec": None}
def hook(module, a, output):
    if st["vec"] is not None:
        h = output[0] if isinstance(output, tuple) else output; p = st["p"]; h[:, p - 1] = (h[:, p - 1].float() + st["vec"]).to(h.dtype)
model.model.layers[LAYER].register_forward_hook(hook)
ids = {int(i): tok(build_prompt(rows[i]["question"]))["input_ids"] for i in tgt}; by_len = collections.defaultdict(list)
for i in tgt: by_len[len(ids[int(i)])].append(int(i))
out = open("analysis/6_steering/results_caa_all.jsonl", "w", encoding="utf-8")
for cname, c in (("base", 0.), ("B c0.5", .5), ("B c1", 1.)):
    tc = time.time()
    for L, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), BS):
            ix = idxs[b0:b0 + BS]; inp = torch.tensor([ids[i] for i in ix], device="cuda"); st["p"] = L
            st["vec"] = None if c == 0 else torch.from_numpy(np.stack([B[grp[i]] * c * norm for i in ix])).float().cuda()
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                             mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, i in enumerate(ix):
                gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]; ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
                ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]}); t3 = "correct" if ok else classify(ans, r["question"], r["relation"], has_cand)[0]
                out.write(json.dumps({"cond": cname, "row": i, "fold": int(grp[i]), "k": int(k[i]), "old_type": ty[i], "answer": ans, "new_type": t3}, ensure_ascii=False) + "\n")
        out.flush()
    print(f"  {cname}: {time.time() - tc:.0f}s", flush=True)
sys.stdout.flush(); os._exit(0)
