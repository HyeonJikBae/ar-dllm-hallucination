"""Upper bound for editing the prompt-end token: overwrite the layer-20 state of a broken row's prompt-end token with the state of another question.   Dream-v0-Base-7B.
Targets: the 310 broken rows of facts 1/5..4/5.   Donor state = the saved step-0 prompt-end layer-20 state of the donor row (no answer written yet, so no answer leaks in).
Conditions:  base | same fact, CORRECT donor, forwards 0-7 | same fact, CORRECT donor, all 32 forwards | OTHER fact (same relation), CORRECT donor, all 32 |
             same fact, ENTITY-wrong donor, all 32 (only targets whose fact has an entity row) | OTHER fact (same relation), BROKEN donor, all 32
New answers are graded / classified with the repo rules.  Usage: CUDA_VISIBLE_DEVICES=2 python3 analysis/6_steering/patch_upper.py"""
import collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import wd
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
from classify_rules import classify
D = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS, BS = 20, 32, 32, 16
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); rel = np.array([r["relation"] for r in rows])
H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r"); rng = np.random.default_rng(0)
tgt = np.where((k >= 1) & (k <= 4) & (ty == "broken"))[0]
byfact = collections.defaultdict(lambda: collections.defaultdict(list))
for i in range(len(rows)): byfact[fid[i]][ty[i]].append(i)
corr_by_rel = collections.defaultdict(list); brk_by_rel = collections.defaultdict(list)
for i in np.where(ty == "correct")[0]: corr_by_rel[rel[i]].append(i)
for i in np.where(ty == "broken")[0]: brk_by_rel[rel[i]].append(i)
def other(pool, f):
    for _ in range(100):
        j = int(rng.choice(pool))
        if fid[j] != f: return j
    return None
DON = {"same fact CORRECT donor, forwards 0-7": ({int(i): int(byfact[fid[i]]["correct"][0]) for i in tgt}, 8),
       "same fact CORRECT donor, all 32": ({int(i): int(byfact[fid[i]]["correct"][0]) for i in tgt}, 32),
       "OTHER fact CORRECT donor (same relation), all 32": ({int(i): other(corr_by_rel[rel[i]], fid[i]) for i in tgt}, 32),
       "same fact ENTITY donor, all 32": ({int(i): int(byfact[fid[i]]["entity"][0]) for i in tgt if byfact[fid[i]]["entity"]}, 32),
       "OTHER fact BROKEN donor (same relation), all 32": ({int(i): other(brk_by_rel[rel[i]], fid[i]) for i in tgt}, 32)}
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id; names = wd._cache.get("names", {}); has_cand = lambda n: (bool(names.get(n)) if n in names else None)
st = {"step": -1, "p": 0, "vec": None, "mask": None, "T": 0}
def hook(module, a, output):
    st["step"] += 1
    if st["vec"] is not None and st["step"] < st["T"]:
        h = output[0] if isinstance(output, tuple) else output; p = st["p"]
        h[:, p - 1] = torch.where(st["mask"][:, None], st["vec"], h[:, p - 1])
model.model.layers[LAYER].register_forward_hook(hook)
ids = {int(i): tok(build_prompt(rows[i]["question"]))["input_ids"] for i in tgt}; by_len = collections.defaultdict(list)
for i in tgt: by_len[len(ids[int(i)])].append(int(i))
R = collections.defaultdict(dict); A = collections.defaultdict(dict); out = open("analysis/6_steering/results_patch.jsonl", "w", encoding="utf-8")
for cname, (don, T) in [("base", ({}, 0))] + list(DON.items()):
    tc = time.time(); st["T"] = T
    for L, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), BS):
            ix = idxs[b0:b0 + BS]; inp = torch.tensor([ids[i] for i in ix], device="cuda"); st["step"] = -1; st["p"] = L
            if cname == "base": st["vec"] = None
            else:
                st["mask"] = torch.tensor([don.get(i) is not None for i in ix], device="cuda")
                st["vec"] = torch.from_numpy(np.stack([np.asarray(H[don[i], 0], dtype=np.float32) if don.get(i) is not None else np.zeros(3584, np.float32) for i in ix])).to("cuda", torch.bfloat16)
            with torch.inference_mode():
                o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                             mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, i in enumerate(ix):
                gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]; ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
                ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]}); t3 = "correct" if ok else classify(ans, r["question"], r["relation"], has_cand)[0]
                R[cname][i] = t3; A[cname][i] = ans; d = don.get(i)
                out.write(json.dumps({"cond": cname, "row": i, "donor": d, "k": int(k[i]), "answer": ans, "new_type": t3, "donor_answer": rows[d]["model_answer"] if d is not None else None}, ensure_ascii=False) + "\n")
    out.flush(); print(f"  {cname}: {time.time() - tc:.0f}s", flush=True)
print("\n==== broken rows of facts 1/5..4/5 -> new answer type (rows that have a donor in that condition)")
for cname, (don, T) in [("base", ({int(i): 0 for i in tgt}, 0))] + list(DON.items()):
    rr = [int(i) for i in tgt if don.get(int(i)) is not None]; c = collections.Counter(R[cname][i] for i in rr); n = len(rr)
    base_sub = collections.Counter(R["base"][i] for i in rr)
    same_as_donor = np.mean([A[cname][i] == rows[don[i]]["model_answer"] for i in rr]) if cname != "base" else float("nan")
    print(f"  {cname:<48s} n={n:>3d} correct {c['correct']/n:.3f} entity {c['entity']/n:.3f} broken {c['broken']/n:.3f} | base on the same rows: correct {base_sub['correct']/n:.3f} broken {base_sub['broken']/n:.3f} | answer = donor's answer {same_as_donor:.3f}")
sys.stdout.flush(); os._exit(0)
