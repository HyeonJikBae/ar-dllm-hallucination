"""Re-run the cross-fitted broken-signal test with the ORIGINAL batch shape (64 rows of the same prompt length, no padding), so that "no intervention" can reproduce the stored answers.
Target rows (the broken rows of facts 1/5..4/5 of both folds, same signals as crossfit_broken.py) are packed by prompt length into batches of 64; empty places are filled with other rows of the same prompt length (filler results are discarded, fillers are never edited).
Step 1  base only: agreement with the stored answers.   If the agreement is below --min-agree, the pipeline stops and says so.
Step 2  conditions: remove BROKEN signal | add BROKEN signal a1 | add CORRECT signal a1 | remove random latents #0..#2 (the same random draws as crossfit_broken.py).
Step 3  writes analysis/6_steering/pad64_summary.txt (pooled over both folds, next to the batch-16 results of crossfit_broken.py).  Resumable: finished conditions are read back from results_pad64.jsonl.
Usage: CUDA_VISIBLE_DEVICES=1 python3 analysis/6_steering/pad64_pipeline.py [--min-agree 0.97]"""
import argparse, collections, json, os, sys, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "analysis"); sys.path.insert(0, "data/1_raw"); sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
import sae_io, wd
from run_inference import build_prompt, extract_answer
from run_grading import is_correct
from classify_rules import classify
ap = argparse.ArgumentParser(); ap.add_argument("--min-agree", type=float, default=0.97); ap.add_argument("--T", type=int, default=8); ap.add_argument("--K", type=int, default=20); ap.add_argument("--draws", type=int, default=3)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="analysis/6_steering/results_pad64.jsonl"); ap.add_argument("--summary", default="analysis/6_steering/pad64_summary.txt")
args = ap.parse_args(); K = args.K
D = "data"; MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; LAYER, CANVAS, STEPS, B = 20, 32, 32, 64
rows = [json.loads(l) for l in open(f"{D}/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
plen = np.array([json.loads(l)["prompt_len"] for l in open(f"{D}/1_raw/dream_hidden_steps/meta.jsonl")])
stored = [json.loads(l)["answer"] for l in sorted((json.loads(l) for l in open(f"{D}/1_raw/dream_hidden_steps/answers.jsonl", encoding="utf-8")), key=lambda d: d["i"]) for _ in [0]] if False else None
popqa = {str(json.loads(l)["id"]): json.loads(l)["s_uri"] for l in open(f"{D}/1_raw/popqa_raw.jsonl", encoding="utf-8")}
sub = np.array([popqa[r["sample_id"]] for r in rows]); rng = np.random.default_rng(0); subs = rng.permutation(np.unique(sub)); half = np.isin(sub, subs[: len(subs) // 2])
ty = np.array([r["type3"] if not r["correct"] else "correct" for r in rows]); k_all = np.array([r["k_correct"] for r in rows]); fid = np.array([r["sample_id"] for r in rows]); mid = (k_all >= 1) & (k_all <= 4)
Wd = sae_io._load(sae_io.DREAM); enc = sae_io._encoder({a: b.cuda() for a, b in Wd.items()}, json.load(open(f"{sae_io.DREAM}/cfg.json")).get("apply_b_dec_to_input", True))
Wu = Wd["W_dec"] / Wd["W_dec"].norm(dim=1, keepdim=True); H = np.load(f"{D}/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
encode = lambda idx: torch.cat([enc(torch.from_numpy(np.stack([np.asarray(H[i, 0], dtype=np.float32) for i in idx[b:b + 512]])).cuda()).cpu() for b in range(0, len(idx), 512)]).numpy()
def find_signal(src):                                         # identical to crossfit_broken.py
    idx = np.where(src & mid & np.isin(ty, ["correct", "broken", "entity"]))[0]; A = encode(idx); on = (A > 0).astype(np.float32); t = ty[idx]; f = fid[idx]; kk = k_all[idx]
    byf = collections.defaultdict(lambda: {"idx": collections.defaultdict(list), "k": 0})
    for j, (ff, tt, k_) in enumerate(zip(f, t, kk)): byf[ff]["idx"][tt].append(j); byf[ff]["k"] = k_
    rate = lambda f_, tt: on[byf[f_]["idx"][tt]].mean(0)
    d = {g: [] for g in (1, 2, 3, 4)}; e, ce = [], []
    for ff, v in byf.items():
        if v["idx"]["correct"] and v["idx"]["broken"]: d[v["k"]].append(rate(ff, "correct") - rate(ff, "broken"))
        if v["idx"]["entity"] and v["idx"]["broken"]: e.append(rate(ff, "entity") - rate(ff, "broken"))
        if v["idx"]["correct"] and v["idx"]["entity"]: ce.append(rate(ff, "correct") - rate(ff, "entity"))
    gc = [np.mean(d[g], 0) for g in d if d[g]] + [np.mean(ce, 0)]; gb = [-np.mean(d[g], 0) for g in d if d[g]] + [-np.mean(e, 0)]
    sc_c, sc_b = np.min(gc, 0), np.min(gb, 0); allrate = on.mean(0)
    okc = (allrate <= 0.5) & (on[t == "correct"].mean(0) >= 0.05); okb = (allrate <= 0.5) & (on[t == "broken"].mean(0) >= 0.05)
    return np.argsort(-np.where(okc, sc_c, -9))[:K], np.argsort(-np.where(okb, sc_b, -9))[:K], allrate
norm = float(np.mean([np.linalg.norm(np.asarray(H[i, 0], dtype=np.float32)) for i in np.random.default_rng(1).choice(len(rows), 500, replace=False)]))
basis = lambda lat: torch.linalg.qr(Wu[lat].T.float())[0].cuda()
addv = lambda lat, a: (lambda v: (a * norm * v / v.norm()).cuda())(Wu[lat].sum(0).float())
FOLDS = []; targets = []                                       # fold 0: signal from half 1, steer half 2;  fold 1: the reverse
for fi, (src, tgt) in enumerate(((half, ~half), (~half, half))):
    up, dn, allrate = find_signal(src); cand = np.where(allrate > 0.01)[0]; qs = np.quantile(allrate[cand], np.linspace(0, 1, 11)); dec = np.clip(np.digitize(allrate, qs[1:-1]), 0, 9)
    rr = np.random.default_rng(5); draws = []
    for _ in range(args.draws):
        pl = []
        for j in dn:
            pool = [c for c in cand if dec[c] == dec[j] and c not in set(dn) and c not in set(up) and c not in pl]; pl.append(int(rr.choice(pool)))
        draws.append(pl)
    FOLDS.append({"remove BROKEN signal": ("proj", basis(dn)), "add BROKEN signal a1": ("add", addv(dn, 1.)), "add CORRECT signal a1": ("add", addv(up, 1.)),
                  **{f"remove random latents #{q}": ("proj", basis(l)) for q, l in enumerate(draws)}})
    targets += [(int(i), fi) for i in np.where(tgt & mid & (ty == "broken"))[0]][: (args.limit or 10**9)]
COND_NAMES = ["base", "remove BROKEN signal", "add BROKEN signal a1", "add CORRECT signal a1"] + [f"remove random latents #{q}" for q in range(args.draws)]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
SP = set(tok.all_special_ids); mask_id = tok.mask_token_id; names = wd._cache.get("names", {}); has_cand = lambda n: (bool(names.get(n)) if n in names else None)
st = {"step": -1, "p": 0, "cond": None, "fold": None}
def hook(module, a, output):
    st["step"] += 1
    if st["cond"] is not None and st["step"] < args.T:
        h = output[0] if isinstance(output, tuple) else output; p = st["p"]; x = h[:, p - 1].float(); y = x.clone()
        for fi in (0, 1):
            m = (st["fold"] == fi)
            if not m.any(): continue
            kind, v = FOLDS[fi][st["cond"]]; yy = x - (x @ v) @ v.T if kind == "proj" else x + v[None]; y = torch.where(m[:, None], yy, y)
        h[:, p - 1] = y.to(h.dtype)
model.model.layers[LAYER].register_forward_hook(hook)
bylen = collections.defaultdict(list)
for i, fi in targets: bylen[int(plen[i])].append((i, fi))
tset = {i for i, _ in targets}; batches = []
for L, lst in sorted(bylen.items()):
    fill = [int(i) for i in np.where(plen == L)[0] if int(i) not in tset]
    for b0 in range(0, len(lst), B):
        chunk = lst[b0:b0 + B]; batches.append((L, chunk, fill[: B - len(chunk)]))
print(f"target rows {len(targets)}, batches of {B}: {len(batches)}", flush=True)
done = collections.defaultdict(dict)
if os.path.exists(args.out):
    for l in open(args.out, encoding="utf-8"): r = json.loads(l); done[r["cond"]][r["row"]] = r
out_f = open(args.out, "a", encoding="utf-8")
def run_cond(cname):
    if len(done.get(cname, {})) == len(targets): return
    t0 = time.time(); st["cond"] = None if cname == "base" else cname
    for L, chunk, fill in batches:
        ids = [i for i, _ in chunk] + fill; inp = torch.tensor([tok(build_prompt(rows[i]["question"]))["input_ids"] for i in ids], device="cuda")
        st["fold"] = torch.tensor([fi for _, fi in chunk] + [-1] * len(fill), device="cuda"); st["step"] = -1; st["p"] = L
        with torch.inference_mode():
            o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                         mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j, (i, fi) in enumerate(chunk):
            gen = o[j][L:].tolist(); gen = gen[: next((q for q, t in enumerate(gen) if t in SP), len(gen))]; ans = extract_answer(tok.decode(gen, skip_special_tokens=True)); r = rows[i]
            ok = is_correct({"model_answer": ans, "gold_value": r["gold_value"], "gold_aliases": r["gold_aliases"]}); t3 = "correct" if ok else classify(ans, r["question"], r["relation"], has_cand)[0]
            rec = {"cond": cname, "row": i, "fold": fi, "k": int(k_all[i]), "rule": r["rule"], "answer": ans, "new_type": t3, "stored_answer": r["model_answer"]}; done[cname][i] = rec; out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out_f.flush(); print(f"[{cname}] {time.time() - t0:.0f}s", flush=True)
def write(lines):
    open(args.summary, "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)
run_cond("base")
ag = np.mean([done["base"][i]["answer"] == done["base"][i]["stored_answer"] for i, _ in targets]); agt = np.mean([done["base"][i]["new_type"] == "broken" for i, _ in targets])
head = [f"STEP 1: no-intervention run with the original batch shape: answer identical to the stored answer in {ag:.3f} of {len(targets)} broken rows; still broken in {agt:.3f} (batch-16 runs gave 0.86 identical / 0.873 broken)."]
if ag < args.min_agree: write(head + [f"STOPPED: agreement below {args.min_agree}; steps 2 and 3 were not run."]); os._exit(0)
for c in COND_NAMES[1:]: run_cond(c)
cf = collections.defaultdict(dict)                               # batch-16 results of crossfit_broken.py, same rows
if os.path.exists("analysis/6_steering/results_crossfit.jsonl"):
    for l in open("analysis/6_steering/results_crossfit.jsonl", encoding="utf-8"): r = json.loads(l); cf[r["cond"]][r["row"]] = r["new_type"]
lines = head + ["", "STEP 2/3: broken rows of facts 1/5..4/5, both folds pooled. New answer types: correct / entity / broken (original batch shape, then batch 16 of crossfit_broken.py)"]
for c in COND_NAMES:
    cnt = collections.Counter(done[c][i]["new_type"] for i, _ in targets); n = len(targets); s = f"  {c:<28s} batch64: correct {cnt['correct']/n:.3f} entity {cnt['entity']/n:.3f} broken {cnt['broken']/n:.3f}"
    if cf.get(c): cc = collections.Counter(cf[c][i] for i, _ in targets if i in cf[c]); m = sum(cc.values()); s += f" | batch16: correct {cc['correct']/m:.3f} entity {cc['entity']/m:.3f} broken {cc['broken']/m:.3f}"
    lines.append(s)
rb = np.array([np.mean([done[c][i]["new_type"] == "broken" for i, _ in targets]) for c in COND_NAMES if c.startswith("remove random")]); lines.append(f"random-latent controls (batch64): broken rate mean {rb.mean():.3f}, min {rb.min():.3f}, max {rb.max():.3f}")
for kk_ in (4, 3, 2, 1):
    ii = [i for i, _ in targets if k_all[i] == kk_]; line = f"  k={kk_}/5 (n={len(ii)}) broken rate: " + ", ".join(f"{c.split(' ')[0]+' '+c.split(' ')[1] if c!='base' else 'base'}: {np.mean([done[c][i]['new_type']=='broken' for i in ii]):.2f}" for c in COND_NAMES[:4]); lines.append(line)
write(lines); os._exit(0)
