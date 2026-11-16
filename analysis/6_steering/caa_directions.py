"""Which mean-difference direction works best, and on which kind of row?   Dream-v0-Base-7B, layer 20, prompt-end token, all 32 forwards, c * mean||h||.
Directions, each from the other 4 subject folds, within the same fact (facts 1/5..4/5), averaged over facts, unit length:
  B   = correct - broken          E = correct - entity          BE = unit(B + E)          C = entity - broken
  D   = broken rows split at random into two halves, mean difference (meaningless direction), 4 draws
Targets (facts 1/5..4/5): all 310 broken rows, 300 random entity-wrong rows, 300 random correct rows.
Conditions: base | B, E, BE, C at c 0.5 and 1 | D x4 at c 1.
Usage: CUDA_VISIBLE_DEVICES=2 python3 analysis/6_steering/caa_directions.py"""
import collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import wd
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
from classify_rules import classify
D_ = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS, BS, NF, ND = 20, 32, 32, 16, 5, 4
rows = [json.loads(l) for l in open(f"{D_}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D_}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); order = {x: j for j, x in enumerate(subs)}; grp = np.array([order[x] % NF for x in sub])
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k >= 1) & (k <= 4)
H = np.load(f"{D_}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r"); X = np.asarray(H[:, 0], dtype=np.float32); norm = float(np.linalg.norm(X[np.random.default_rng(1).choice(len(X), 500, replace=False)], axis=1).mean())
unit = lambda v: v / np.linalg.norm(v); dirs = {}
for g in range(NF):
    tr = grp != g; byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & np.isin(ty, ["correct", "broken", "entity"]))[0]: byf[fid[i]][ty[i]].append(i)
    wf = lambda a, b: (unit(np.mean([X[v[a]].mean(0) - X[v[b]].mean(0) for v in byf.values() if v[a] and v[b]], 0)), sum(1 for v in byf.values() if v[a] and v[b]))
    (B, nb), (E, ne), (C, nc) = wf("correct", "broken"), wf("correct", "entity"), wf("entity", "broken")
    dirs[g] = {"B": B, "E": E, "BE": unit(B + E), "C": C}
    bi = np.where(tr & (ty == "broken"))[0]
    for q in range(ND):
        p = np.random.default_rng(100 * q + g).permutation(bi); dirs[g][f"D{q}"] = unit(X[p[: len(p) // 2]].mean(0) - X[p[len(p) // 2:]].mean(0))
    if g == 0: print(f"fold 0 facts used: B {nb}, E {ne}, C {nc} | cosines: B-E {B @ E:.2f}, B-C {B @ C:.2f}, E-C {E @ C:.2f}, BE-B {dirs[0]['BE'] @ B:.2f}, BE-E {dirs[0]['BE'] @ E:.2f}", flush=True)
r5 = np.random.default_rng(5)
G = {"broken": np.where(mid & (ty == "broken"))[0], "entity": r5.permutation(np.where(mid & (ty == "entity"))[0])[:300], "correct": r5.permutation(np.where(mid & (ty == "correct"))[0])[:300]}
tgt = np.concatenate(list(G.values()))
CONDS = [("base", None, 0.)] + [(f"{d} c{c}", d, c) for d in ("B", "E", "BE", "C") for c in (.5, 1.)] + [(f"D{q} c1", f"D{q}", 1.) for q in range(ND)]
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
R = collections.defaultdict(dict); out = open("analysis/6_steering/results_caa_directions.jsonl", "w", encoding="utf-8")
for cname, d, c in CONDS:
    tc = time.time()
    for L, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), BS):
            ix = idxs[b0:b0 + BS]; inp = torch.tensor([ids[i] for i in ix], device="cuda"); st["p"] = L
            st["vec"] = None if d is None else torch.from_numpy(np.stack([dirs[grp[i]][d] * c * norm for i in ix])).float().cuda()
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                             mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, i in enumerate(ix):
                gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]; ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
                ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]}); t3 = "correct" if ok else classify(ans, r["question"], r["relation"], has_cand)[0]
                R[cname][i] = t3; out.write(json.dumps({"cond": cname, "row": i, "fold": int(grp[i]), "k": int(k[i]), "old_type": ty[i], "answer": ans, "new_type": t3}, ensure_ascii=False) + "\n")
    out.flush(); print(f"  {cname}: {time.time() - tc:.0f}s", flush=True)
sh = lambda c, g, t: np.mean([R[c][int(i)] == t for i in G[g]])
for g in G:
    print(f"\n==== originally {g.upper()} rows (n={len(G[g])}): correct / entity / broken")
    for cname, d, c in CONDS:
        if d is not None and d.startswith("D"): continue
        print(f"  {cname:<8s} correct {sh(cname, g, 'correct'):.3f} | entity {sh(cname, g, 'entity'):.3f} | broken {sh(cname, g, 'broken'):.3f}")
    dc = np.array([sh(f"D{q} c1", g, "correct") for q in range(ND)]); db = np.array([sh(f"D{q} c1", g, "broken") for q in range(ND)])
    print(f"  D x{ND} c1: correct mean {dc.mean():.3f} ({dc.min():.3f}-{dc.max():.3f}) | broken mean {db.mean():.3f} ({db.min():.3f}-{db.max():.3f})")
sys.stdout.flush(); os._exit(0)
