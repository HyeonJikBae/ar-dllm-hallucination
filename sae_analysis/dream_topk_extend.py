"""Extend the Dream topk_margin run to more PopQA facts (outside the 1,000-fact sample): same prompt, 32-token canvas, 32 steps,
temperature 0, alg=topk_margin. Writes rows.jsonl (inputs) and topk_margin_s32.jsonl (answers) under sae_analysis/results_twelve/dream_ext/.
Usage: CUDA_VISIBLE_DEVICES=0 /home/minyeol/.conda/envs/sae-dream/bin/python sae_analysis/dream_topk_extend.py --n-facts 4000
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "sae")):
    sys.path.insert(0, p)
OUTD = "sae_analysis/results_twelve/dream_ext"
CANVAS, BATCH, STEPS = 32, 64, 32

def build_rows(n_facts, seed):
    sample = {json.loads(l)["sample_id"] for l in open("analysis/graded/sampled_1000_seed42/dream-7b_popqa_graded.jsonl")}
    raw = {}
    for l in open("data/popqa/1. popqa_raw.jsonl"):
        x = json.loads(l); raw[str(x["id"])] = x["prop"]
    by = {}
    for l in open("analysis/graded/1. reviewed 1/dream-7b_popqa_graded.jsonl"):
        r = json.loads(l); by.setdefault(r["sample_id"], []).append(r)
    facts = sorted(f for f, rs in by.items() if f not in sample and len(rs) == 5 and not any(r.get("gold_issue") for r in rs))
    chosen = np.random.default_rng(seed).permutation(facts)[:n_facts]
    rows = []
    for f in chosen:
        for r in sorted(by[f], key=lambda r: ["question", "v1", "v2", "v3", "v4"].index(r["version"])):
            rows.append({"sample_id": f, "version": r["version"], "question": r["question"], "gold_value": r["gold_value"],
                         "gold_aliases": r["gold_aliases"], "relation": raw[f], "gold_issue": False})
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-facts", type=int, default=4000); ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    import torch
    from dream_acts import load_dream
    from run_inference import build_prompt, extract_answer
    os.makedirs(OUTD, exist_ok=True)
    rows_path = f"{OUTD}/rows.jsonl"
    if not os.path.exists(rows_path):
        with open(rows_path, "w", encoding="utf-8") as f:
            for r in build_rows(a.n_facts, a.seed):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [json.loads(l) for l in open(rows_path, encoding="utf-8")]
    path = f"{OUTD}/topk_margin_s32.jsonl"
    done = set()
    if os.path.exists(path):
        done = {(d["sample_id"], d["version"]) for d in map(json.loads, open(path)) if d["sample_id"]}
    todo = [r for r in rows if (r["sample_id"], r["version"]) not in done]
    print(f"rows {len(rows)} (facts {len(rows) // 5}), to generate {len(todo)}", flush=True)
    model, tok = load_dream("cuda"); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    with open(path, "a", encoding="utf-8") as fo:
        for b in range(0, len(todo), BATCH):
            batch = todo[b:b + BATCH]
            enc = tok([build_prompt(r["question"]) for r in batch], return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.inference_mode():
                out = model.diffusion_generate(inputs=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=CANVAS, steps=STEPS,
                    temperature=0.0, alg="topk_margin", alg_temp=0.0, mask_token_id=tok.mask_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for r, o in zip(batch, out):
                gen = o[enc["input_ids"].shape[1]:]; raw_t = tok.decode(gen, skip_special_tokens=True)
                fo.write(json.dumps({"sample_id": r["sample_id"], "version": r["version"], "answer": extract_answer(raw_t), "raw": raw_t,
                                     "has_eos": bool((gen == tok.eos_token_id).any())}, ensure_ascii=False) + "\n")
            fo.flush()
            if (b // BATCH) % 25 == 0:
                print(f"  {b + len(batch)}/{len(todo)}", flush=True)
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
