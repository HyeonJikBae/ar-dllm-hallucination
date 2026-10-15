"""Steering experiment: add alpha * sigma * (unit direction) to Dream's layer-20 output at the last prompt token, at every denoising step,
and regenerate (topk_margin, 32 steps, T=0). Direction from steer_directions.py (fitted on the OTHER subject folds: no test leakage).
Conditions: baseline (alpha=0), dim +1/+2/+4/-2, sae +2/+4, random +2/+4.   Subset: 1,000 facts x 5 wordings.
Output: sae_analysis/results_twelve/steer/<cond>.jsonl (resumable) and report_steer.txt.
Usage: CUDA_VISIBLE_DEVICES=0 /home/minyeol/.conda/envs/sae-dream/bin/python sae_analysis/dream_steer.py [--smoke]
"""
import argparse, collections, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "grounded_dream"), os.path.join(ROOT, "sae")):
    sys.path.insert(0, p)
from string_grader import contains_gold, string_rule  # noqa: E402
OUT = "sae_analysis/results_twelve/steer"; ROWS = "sae_analysis/results_twelve/dream_ext/rows.jsonl"
CONDS = [("baseline", None, 0.0), ("dim_p1", 0, 1.0), ("dim_p2", 0, 2.0), ("dim_p4", 0, 4.0), ("dim_m2", 0, -2.0),
         ("sae_p2", 1, 2.0), ("sae_p4", 1, 4.0), ("rand_p2", 2, 2.0), ("rand_p4", 2, 4.0)]

def grade(a, r):
    code = string_rule(a, r["question"], r["gold_value"], r["gold_aliases"], r["relation"])
    return code in ("C-alias", "C-short") or contains_gold(a, r["gold_value"], r["gold_aliases"]), code

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--eval-only", action="store_true"); a = ap.parse_args()
    Z = np.load(f"{OUT}/directions.npz", allow_pickle=True); dirs, sigma, subset, facts, ffold = Z["dirs"], Z["sigma"], set(Z["subset"]), Z["facts"], Z["ffold"]
    fold_of = {f: int(k) for f, k in zip(facts, ffold)}
    rows = [r for r in map(json.loads, open(ROWS, encoding="utf-8")) if r["sample_id"] in subset]
    if a.smoke: rows = rows[:128]
    odir = OUT + ("/smoke" if a.smoke else ""); os.makedirs(odir, exist_ok=True)
    if not a.eval_only:
        import torch
        from dream_acts import load_dream
        from run_inference import build_prompt, extract_answer
        model, tok = load_dream("cuda"); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        layer = model.model.layers[20]; state = {"vec": None}
        def hook(mod, inp, out):
            if state["vec"] is not None:
                h = out[0] if isinstance(out, tuple) else out
                h[:, -33, :] += state["vec"]      # last prompt token (left padding + 32-token canvas)
        layer.register_forward_hook(hook)
        for name, d, alpha in CONDS:
            path = f"{odir}/{name}.jsonl"
            done = {(r["sample_id"], r["version"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
            with open(path, "a", encoding="utf-8") as fo:
                for k in range(5):
                    todo = [r for r in rows if fold_of[r["sample_id"]] == k and (r["sample_id"], r["version"]) not in done]
                    state["vec"] = None if d is None else torch.tensor(alpha * sigma[k, 0] * dirs[k, d], dtype=torch.bfloat16, device="cuda")  # same vector norm for dim / sae / random
                    for b in range(0, len(todo), 64):
                        batch = todo[b:b + 64]
                        enc = tok([build_prompt(r["question"]) for r in batch], return_tensors="pt", padding=True, truncation=True).to("cuda")
                        with torch.inference_mode():
                            out = model.diffusion_generate(inputs=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=32, steps=32,
                                temperature=0.0, alg="topk_margin", alg_temp=0.0, mask_token_id=tok.mask_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                        for r, o in zip(batch, out):
                            fo.write(json.dumps({"sample_id": r["sample_id"], "version": r["version"], "answer": extract_answer(tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True))}, ensure_ascii=False) + "\n")
                        fo.flush()
            print(f"{name} done", flush=True)
    # ---- evaluation ----
    key = lambda r: (r["sample_id"], r["version"]); R = {key(r): r for r in rows}
    res = {}
    for name, _, _ in CONDS:
        p = f"{odir}/{name}.jsonl"
        if os.path.exists(p):
            res[name] = {key(d): d["answer"] for d in map(json.loads, open(p))}
    base = res.get("baseline", {}); ks = [k for k in R if all(k in v for v in res.values())]
    yb = {k: grade(base[k], R[k])[0] for k in ks}
    ysum = collections.defaultdict(int)
    for k in ks: ysum[k[0]] += yb[k]
    mixed_f = {f for f in ysum}  # placeholder, real mixed set defined below from baseline of all 5 wordings
    ys_all = collections.defaultdict(list)
    for k in ks: ys_all[k[0]].append(yb[k])
    print(f"행 {len(ks)}개 (사실 {len(ys_all)}개), 같은 행끼리 비교. 정답 = 포함 채점")
    print(f"{'조건':<10}{'정답률':>8}{'오답→정답':>10}{'정답→오답':>10}{'순증':>7}{'빈 답':>8}{'baseline과 답 같음':>20}")
    for name, v in res.items():
        yy = {k: grade(v[k], R[k])[0] for k in ks}; codes = [grade(v[k], R[k])[1] for k in ks]
        up = sum((not yb[k]) and yy[k] for k in ks); dn = sum(yb[k] and not yy[k] for k in ks)
        same = np.mean([v[k].strip() == base[k].strip() for k in ks]); empty = np.mean([c == "6-empty" for c in codes])
        print(f"{name:<10}{np.mean(list(yy.values())):>8.1%}{up:>10}{dn:>10}{up - dn:>+7}{empty:>8.1%}{same:>20.1%}")
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
