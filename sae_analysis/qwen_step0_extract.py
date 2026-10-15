"""Qwen2.5-7B counterpart of dream_step0_extract.py, on the same 4,000 extension facts (dream_ext/rows.jsonl).

For every row: prompt (build_prompt) -> layer-20 output at the last prompt token (prompt_last) and the subject's last token,
and the next-token confidence at the last prompt position. Labels: Qwen's stored greedy answers (analysis/inference/qwen2.5-7b_popqa.jsonl),
string-graded with the same rule as Dream ("contain").
Saved in the same layout as features_dream_step0*: conf.npy [N,4], hid_last.npy, hid_subj.npy, meta.jsonl.

Usage: CUDA_VISIBLE_DEVICES=0 /home/minyeol/.conda/envs/sae/bin/python sae_analysis/qwen_step0_extract.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "grounded_dream")):
    sys.path.insert(0, p)
from string_grader import contains_gold, string_rule  # noqa: E402

ROWS = "sae_analysis/results_twelve/dream_ext/rows.jsonl"
ANSWERS = "analysis/inference/qwen2.5-7b_popqa.jsonl"
OUT = "sae_analysis/features_qwen_step0_ext"
BATCH = 32


def main():
    import torch
    from extract_qwen_features import MODEL, REVISION
    from run_inference import build_prompt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(ROWS, encoding="utf-8")]
    ans = {(d["sample_id"], d["version"]): d["model_answer"] for d in map(json.loads, open(ANSWERS, encoding="utf-8"))}
    subj = {}
    for l in open("data/popqa/1. popqa_raw.jsonl", encoding="utf-8"):
        x = json.loads(l); subj[str(x["id"])] = [n for n in [x["subj"]] + json.loads(x["s_aliases"] or "[]") if n]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION); tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True).eval()
    n = len(rows)
    conf = np.zeros((n, 4), np.float32)
    hl = np.zeros((n, 3584), np.float16); hs = np.zeros((n, 3584), np.float16)
    found = np.zeros(n, bool); plen = np.zeros(n, int)
    for b in range(0, n, BATCH):
        batch = rows[b:b + BATCH]
        prompts = [build_prompt(r["question"]) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        last = enc["attention_mask"].sum(1) - 1
        for j, (r, pr) in enumerate(zip(batch, prompts)):
            i = b + j; p = int(last[j]) + 1; plen[i] = p
            lp = torch.log_softmax(out.logits[j, p - 1].float(), -1)
            top = lp.topk(2)
            a = ans[(r["sample_id"], r["version"])] or ""
            first = tok(" " + a.strip(), add_special_tokens=False)["input_ids"][:1] if a.strip() else []
            conf[i] = [top.values[0].item(), (top.values[0] - top.values[1]).item(), -(lp.exp() * lp).sum().item(), lp[first[0]].item() if first else np.nan]
            h = out.hidden_states[21][j]
            hl[i] = h[p - 1].float().cpu().numpy()
            low = pr.lower(); offs = tok(pr, return_offsets_mapping=True)["offset_mapping"]
            for nm in sorted(subj[r["sample_id"]], key=len, reverse=True):
                k0 = low.find(nm.lower())
                if k0 >= 0:
                    s_end = max(k for k, (a0, b0) in enumerate(offs) if a0 < k0 + len(nm) and b0 > k0)
                    hs[i] = h[s_end].float().cpu().numpy(); found[i] = True
                    break
        if (b // BATCH) % 50 == 0:
            print(f"{b}/{n}", flush=True)
    meta = []
    for i, r in enumerate(rows):
        a = ans[(r["sample_id"], r["version"])] or ""
        code = string_rule(a, r["question"], r["gold_value"], r["gold_aliases"], r["relation"])
        meta.append({"sample_id": r["sample_id"], "version": r["version"], "relation": r["relation"], "answer": a, "code": code,
                     "strict": code in ("C-alias", "C-short"),
                     "contain": code in ("C-alias", "C-short") or contains_gold(a, r["gold_value"], r["gold_aliases"]),
                     "gold_issue": bool(r.get("gold_issue")), "subj_found": bool(found[i]), "prompt_len": int(plen[i])})
    np.save(f"{OUT}/conf.npy", conf); np.save(f"{OUT}/hid_last.npy", hl); np.save(f"{OUT}/hid_subj.npy", hs)
    with open(f"{OUT}/meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("done", n, "subject found:", found.mean(), "accuracy(contain):", np.mean([m["contain"] for m in meta]))
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
