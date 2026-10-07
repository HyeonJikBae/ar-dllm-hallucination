"""Raw Dream-v0-Base-7B layer-20 hidden states at q_last / prompt_last, two input variants.

  plain   prompt only (what the SAEs' unmasked positions look like)
  masked  prompt + N <|mask|> tokens, as at step 0 of diffusion_generate (default N=32)

Rows: kept rows ("6. excl 0of5", uncertain dropped) + rows of the 0-of-5 facts ("5. reviewed 5"
minus kept). Positions are the same two prompt tokens as for Qwen. Rows of equal prompt length are
batched, so no padding is needed (attention_mask="full" as in Dream's own generation loop).

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 sae_analysis/extract_dream_hidden.py --dataset popqa
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_qwen_features import MODEL as QWEN, REVISION as QWEN_REVISION, positions  # noqa: E402
from run_inference import build_prompt  # noqa: E402

MODEL = "Dream-org/Dream-v0-Base-7B"
REVISION = "6572adb5535263e4d1a337b56942ba48b6dee2a9"
LAYER = 20
KEPT = "analysis/graded/6. excl 0of5/dream-7b_{ds}_reclassified.jsonl"
FULL = "analysis/graded/5. reviewed 5/dream-7b_{ds}_reclassified.jsonl"


def load_rows(ds):
    kept_rows = [json.loads(l) for l in open(KEPT.format(ds=ds), encoding="utf-8")]
    kept_ids = {r["sample_id"] for r in kept_rows}
    rows = [dict(r, group="kept") for r in kept_rows if r["verdict"] != "uncertain"]
    for line in open(FULL.format(ds=ds), encoding="utf-8"):
        r = json.loads(line)
        if r["sample_id"] not in kept_ids and r["verdict"] != "uncertain":
            rows.append(dict(r, group="zero"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["popqa", "triviaqa"])
    ap.add_argument("--output-dir", default="sae_analysis/features_dream")
    ap.add_argument("--n-masks", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    rows = load_rows(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    # Dream's tokenizer has no offset mapping; it tokenizes like Qwen's, so take the positions from
    # Qwen's fast tokenizer and assert the ids agree for every row.
    qtok = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REVISION)
    enc = []
    for r in rows:
        p = build_prompt(r["question"])
        ids, ql, pl = positions(qtok, r["question"], p)
        assert ids == tok(p)["input_ids"], f"tokenization differs for {r['sample_id']}/{r['version']}"
        enc.append((ids, ql, pl))
    d = model.config.hidden_size
    out = {(v, p): np.zeros((len(rows), d), dtype=np.float32) for v in ("plain", "masked") for p in ("q_last", "prompt_last")}

    by_len = collections.defaultdict(list)
    for i, (ids, _, _) in enumerate(enc):
        by_len[len(ids)].append(i)
    done = 0
    for length, idxs in sorted(by_len.items()):
        for s in range(0, len(idxs), args.batch_size):
            b = idxs[s : s + args.batch_size]
            ids = torch.tensor([enc[i][0] for i in b], device="cuda")
            for variant in ("plain", "masked"):
                x = ids if variant == "plain" else torch.cat(
                    [ids, torch.full((len(b), args.n_masks), tok.mask_token_id, device="cuda")], 1)
                with torch.no_grad():
                    hs = model(x, attention_mask="full", output_hidden_states=True).hidden_states[LAYER + 1]
                for pname, col in (("q_last", 1), ("prompt_last", 2)):
                    pos = torch.tensor([enc[i][col] for i in b], device="cuda")
                    out[(variant, pname)][b] = hs[torch.arange(len(b)), pos].float().cpu().numpy()
            done += len(b)
        print(f"{done}/{len(rows)}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for (variant, pname), arr in out.items():
        np.save(os.path.join(args.output_dir, f"dream_{args.dataset}_{variant}_{pname}_hidden.npy"), arr)
    with open(os.path.join(args.output_dir, f"dream_{args.dataset}_meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in ("sample_id", "version", "question", "model_answer", "verdict", "reason", "gold_issue", "group")}, ensure_ascii=False) + "\n")
    print(f"done {args.dataset}: {len(rows)} rows")


if __name__ == "__main__":
    main()
