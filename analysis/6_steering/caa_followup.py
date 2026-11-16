"""Follow-up of caa_broken.py (same folds, same direction B, same layer / token / 32 forwards, same scaling c * mean||h||).
 1) broken rows of facts 1/5..4/5 (310): B at c 1.5 and 2;  D-type controls (broken rows split at random into two halves, mean difference) x8 draws at c 1
 2) side effects: 150 CORRECT rows and 150 ENTITY-wrong rows of facts 1/5..4/5: base, B c0.5, B c1, D draw #0 c1
Usage: CUDA_VISIBLE_DEVICES=2 python3 analysis/6_steering/caa_followup.py"""
import collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import wd
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
from classify_rules import classify
D = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS, BS, NF, ND = 20, 32, 32, 16, 5, 8
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); order = {x: j for j, x in enumerate(subs)}; grp = np.array([order[x] % NF for x in sub])
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r"); X = np.asarray(H[:, 0], dtype=np.float32); norm = float(np.linalg.norm(X[np.random.default_rng(1).choice(len(X), 500, replace=False)], axis=1).mean())
unit = lambda v: v / np.linalg.norm(v); dirs = {}
for g in range(NF):
    tr = grp != g; bi = np.where(tr & (ty == "broken"))[0]
    byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & np.isin(ty, ["correct", "broken"]))[0]: byf[fid[i]][ty[i]].append(i)
    within = [X[v["correct"]].mean(0) - X[v["broken"]].mean(0) for v in byf.values() if v["correct"] and v["broken"]]
    dirs[g] = {"B": unit(np.mean(within, 0))}
    for q in range(ND):
        p = np.random.default_rng(100 * q + g).permutation(bi); dirs[g][f"D{q}"] = unit(X[p[: len(p) // 2]].mean(0) - X[p[len(p) // 2:]].mean(0))
print("cosine of B with the D draws (fold 0):", [round(float(dirs[0]["B"] @ dirs[0][f"D{q}"]), 2) for q in range(ND)], flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id; names = wd._cache.get("names", {}); has_cand = lambda n: (bool(names.get(n)) if n in names else None)
st = {"step": -1, "p": 0, "vec": None}
def hook(module, a, output):
    st["step"] += 1
    if st["vec"] is not None:
        h = output[0] if isinstance(output, tuple) else output; p = st["p"]; h[:, p - 1] = (h[:, p - 1].float() + st["vec"]).to(h.dtype)
model.model.layers[LAYER].register_forward_hook(hook)
out = open("analysis/6_steering/results_caa_followup.jsonl", "w", encoding="utf-8"); R = collections.defaultdict(dict)
def run(tag, tgt, conds):
    ids = {int(i): tok(build_prompt(rows[i]["question"]))["input_ids"] for i in tgt}; by_len = collections.defaultdict(list)
    for i in tgt: by_len[len(ids[int(i)])].append(int(i))
    for cname, which, c in conds:
        tc = time.time()
        for L, idxs in sorted(by_len.items()):
            for b0 in range(0, len(idxs), BS):
                ix = idxs[b0:b0 + BS]; inp = torch.tensor([ids[i] for i in ix], device="cuda"); st["step"] = -1; st["p"] = L
                st["vec"] = None if which is None else torch.from_numpy(np.stack([dirs[grp[i]][which] * c * norm for i in ix])).float().cuda()
                with torch.inference_mode():
                    o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                                 mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                for j, i in enumerate(ix):
                    gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]; ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
                    ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]}); t3 = "correct" if ok else classify(ans, r["question"], r["relation"], has_cand)[0]
                    R[(tag, cname)][i] = t3; out.write(json.dumps({"set": tag, "cond": cname, "row": i, "fold": int(grp[i]), "k": int(k[i]), "old_type": ty[i], "answer": ans, "new_type": t3}, ensure_ascii=False) + "\n")
        out.flush(); print(f"  [{tag}] {cname}: {time.time() - tc:.0f}s", flush=True)
brk = np.where(mid & (ty == "broken"))[0]
run("broken", brk, [("B c1.5", "B", 1.5), ("B c2", "B", 2.)] + [(f"D{q} c1", f"D{q}", 1.) for q in range(ND)])
r3 = np.random.default_rng(3); side = np.concatenate([r3.permutation(np.where(mid & (ty == "correct"))[0])[:150], r3.permutation(np.where(mid & (ty == "entity"))[0])[:150]])
run("side", side, [("base", None, 0), ("B c0.5", "B", .5), ("B c1", "B", 1.), ("D0 c1", "D0", 1.)])
sh = lambda key, g, t: np.mean([R[key][int(i)] == t for i in g])
print("\n==== broken rows (n=%d)" % len(brk))
for c in ["B c1.5", "B c2"] + [f"D{q} c1" for q in range(ND)]: print(f"  {c:<10s} correct {sh(('broken', c), brk, 'correct'):.3f} | entity {sh(('broken', c), brk, 'entity'):.3f} | broken {sh(('broken', c), brk, 'broken'):.3f}")
dc = np.array([sh(("broken", f"D{q} c1"), brk, "correct") for q in range(ND)]); db = np.array([sh(("broken", f"D{q} c1"), brk, "broken") for q in range(ND)])
print(f"  D draws c1: correct mean {dc.mean():.3f} ({dc.min():.3f}-{dc.max():.3f}) | broken mean {db.mean():.3f} ({db.min():.3f}-{db.max():.3f})")
for name, g in (("CORRECT rows", side[:150]), ("ENTITY rows", side[150:])):
    print(f"\n==== side effects on {name} (n={len(g)})")
    for c in ("base", "B c0.5", "B c1", "D0 c1"): print(f"  {c:<8s} correct {sh(('side', c), g, 'correct'):.3f} | entity {sh(('side', c), g, 'entity'):.3f} | broken {sh(('side', c), g, 'broken'):.3f} | abstain {sh(('side', c), g, 'abstain'):.3f}")
sys.stdout.flush(); os._exit(0)
