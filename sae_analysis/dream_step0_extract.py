"""Dream, before decoding: first-answer-token confidence and layer-20 hidden states for the 1,000-fact PopQA sample.

Input to Dream = prompt tokens + 32 <|mask|> tokens (the state at denoising step 0), no attention padding (batch of 1).
Dream shifts its logits: the first answer token is predicted by the raw logits at the LAST PROMPT position (index p-1).
Saved per row (aligned with meta.jsonl):
  conf.npy       [N, 4]  log p(top1), log p(top1) - log p(top2), entropy (nats), log p(top1 token of the row's own answer)
  hid_last.npy   [N, 3584] float16  layer-20 output at the last prompt token (prompt_last)
  hid_subj.npy   [N, 3584] float16  layer-20 output at the subject's last token (nan row -> subj_found False)
Labels: Dream answers decoded with topk_margin (deterministic) from dream_sampler_compare.py, string-graded.

Usage:  CUDA_VISIBLE_DEVICES=0 /home/minyeol/.conda/envs/sae-dream/bin/python sae_analysis/dream_step0_extract.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "grounded_dream"), os.path.join(ROOT, "sae")):
    sys.path.insert(0, p)
import dream_sampler_compare as DS  # noqa: E402
from string_grader import contains_gold, string_rule  # noqa: E402

OUT = os.environ.get("STEP0_OUT", "sae_analysis/features_dream_step0")
ANSWERS = os.environ.get("STEP0_ANSWERS", "sae_analysis/results_twelve/dream_sampler/topk_margin_s32.jsonl")
ROWS_FILE = os.environ.get("STEP0_ROWS")  # jsonl of rows (sample_id, version, question, gold_value, gold_aliases, relation, gold_issue); default = the 1,000-fact sample
CANVAS = 32


def main():
    import torch
    from dream_acts import DREAM_MASK_TOKEN_ID, load_dream
    from run_inference import build_prompt
    from transformers import AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(ROWS_FILE, encoding="utf-8")] if ROWS_FILE else DS.load_rows()
    ans = {(d["sample_id"], d["version"]): d for d in map(json.loads, open(ANSWERS, encoding="utf-8")) if d["sample_id"]}
    subj = {}
    for l in open("data/popqa/1. popqa_raw.jsonl", encoding="utf-8"):
        x = json.loads(l); subj[str(x["id"])] = [n for n in [x["subj"]] + json.loads(x["s_aliases"] or "[]") if n]
    model, tok = load_dream("cuda")
    tokq = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", revision="d149729398750b98c0af14eb82c78cfe92750796")  # same vocab, fast offsets
    n = len(rows)
    conf = np.zeros((n, 4), np.float32)
    hl = np.zeros((n, 3584), np.float16); hs = np.zeros((n, 3584), np.float16)
    found = np.zeros(n, bool)
    meta = []
    for i, r in enumerate(rows):
        prompt = build_prompt(r["question"])
        enc = tokq(prompt, return_offsets_mapping=True)
        ids = enc["input_ids"]
        if i < 20:
            assert tok(prompt)["input_ids"] == ids, "Dream and Qwen tokenizers disagree"
        p = len(ids)
        low = prompt.lower(); s_end = None
        for nm in sorted(subj[r["sample_id"]], key=len, reverse=True):
            j = low.find(nm.lower())
            if j >= 0:
                s_end = max(k for k, (a, b) in enumerate(enc["offset_mapping"]) if a < j + len(nm) and b > j)
                break
        x = torch.tensor([ids + [DREAM_MASK_TOKEN_ID] * CANVAS], device="cuda")
        with torch.inference_mode():
            out = model(x, use_cache=False, output_hidden_states=True)
        lp = torch.log_softmax(out.logits[0, p - 1].float(), -1)
        top = lp.topk(2)
        a = ans[(r["sample_id"], r["version"])]["answer"]
        first = tokq(" " + a.strip(), add_special_tokens=False)["input_ids"][:1] if a.strip() else []
        conf[i] = [top.values[0].item(), (top.values[0] - top.values[1]).item(), -(lp.exp() * lp).sum().item(), lp[first[0]].item() if first else np.nan]
        h = out.hidden_states[21][0].float().cpu().numpy()
        hl[i] = h[p - 1]
        if s_end is not None:
            hs[i] = h[s_end]; found[i] = True
        code = string_rule(a, r["question"], r["gold_value"], r["gold_aliases"], r["relation"])
        meta.append({"sample_id": r["sample_id"], "version": r["version"], "relation": r["relation"], "answer": a, "code": code,
                     "strict": code in ("C-alias", "C-short"),
                     "contain": code in ("C-alias", "C-short") or contains_gold(a, r["gold_value"], r["gold_aliases"]),
                     "gold_issue": bool(r.get("gold_issue")), "subj_found": bool(found[i]), "prompt_len": p})
        if i % 500 == 0:
            print(f"{i}/{n}", flush=True)
    np.save(f"{OUT}/conf.npy", conf); np.save(f"{OUT}/hid_last.npy", hl); np.save(f"{OUT}/hid_subj.npy", hs)
    with open(f"{OUT}/meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("done", n, "subject found:", found.mean())
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
