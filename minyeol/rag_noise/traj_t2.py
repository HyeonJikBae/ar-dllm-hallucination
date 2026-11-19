"""PREREG_traj.md T2: Dream 300 질의 x gold/hard3/rand3 x 샘플러, step 별 argmax / top-1 확률 / 가려짐 기록 (bs1).
usage: CUDA_VISIBLE_DEVICES=N python traj_t2.py --sampler entropy:16|topk_margin:16|entropy:8   -> results/traj_t2_<alg>_s<steps>.{jsonl,npz}"""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gen import prompt, cond_docs, first_line, MODELS
from transformers import AutoTokenizer, AutoModel
ap = argparse.ArgumentParser(); ap.add_argument("--sampler", required=True); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
alg, steps = a.sampler.split(":"); steps = int(steps); MAXN = 16
rows = [json.loads(l) for l in open(f"{HERE}/data/musique_anchor.jsonl")]
pick = sorted(np.random.default_rng(2611).permutation(len(rows))[:300].tolist()); rows = [rows[i] for i in pick][: a.limit or None]
shots = json.load(open(f"{HERE}/data/fewshot.json"))[:2]
tok = AutoTokenizer.from_pretrained(MODELS["dream"], trust_remote_code=True); tok.padding_side = "left"
model = AutoModel.from_pretrained(MODELS["dream"], torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
name = f"traj_t2_{alg}_s{steps}" + ("_pilot" if a.limit else ""); jobs = [(r, c) for r in rows for c in ("gold", "hard3", "rand3")]
AR, PR, MK, FN, out = [], [], [], [], []; t0 = time.perf_counter()
for n_, (r, c) in enumerate(jobs):
    x = tok([prompt(r["question"], cond_docs(r, c), shots)], return_tensors="pt").to("cuda"); tr = []
    def hook(i, xx, logits):
        lg = logits[:, -MAXN:].float(); p = torch.softmax(lg, -1); pm, am = p.max(-1)
        tr.append((am[0].int().cpu(), pm[0].cpu(), (xx[0, -MAXN:] == tok.mask_token_id).cpu())); return logits
    with torch.inference_mode():
        o = model.diffusion_generate(inputs=x["input_ids"], attention_mask=x["attention_mask"], max_new_tokens=MAXN, steps=steps, temperature=0.0, alg=alg,
                                     mask_token_id=tok.mask_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id, generation_logits_hook_func=hook)
    o = o.sequences if hasattr(o, "sequences") else o
    AR.append(torch.stack([t[0] for t in tr]).numpy()); PR.append(torch.stack([t[1] for t in tr]).numpy().astype(np.float16)); MK.append(torch.stack([t[2] for t in tr]).numpy())
    fin = o[0, -MAXN:].int().cpu().numpy(); FN.append(fin); g = tok.decode(fin, skip_special_tokens=True)
    out.append(dict(qid=r["qid"], qi=r["qi"], cond=c, raw=g, pred=first_line(g), answers=r["answers"]))
    if n_ % 100 == 0: print(f"  {n_}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
np.savez_compressed(f"{HERE}/results/{name}.npz", argmax=np.stack(AR), prob=np.stack(PR), masked=np.stack(MK), final=np.stack(FN))
with open(f"{HERE}/results/{name}.jsonl", "w") as f:
    for d in out: f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"DONE {name} {time.perf_counter() - t0:.0f}s", flush=True)
