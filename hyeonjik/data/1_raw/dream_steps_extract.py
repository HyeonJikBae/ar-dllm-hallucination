"""Dream denoising-step hidden states for the PopQA rows of "7. final" (answers and hidden states from the SAME run).

Setting (fixed, recorded in config.json): Dream-v0-Base-7B (pinned revision), 32-token canvas, 32 denoising steps,
alg=topk_margin, temperature 0, no padding (rows of equal prompt length are batched), as in dream_topk_extend.py.

At every denoising step i (= i-th forward pass) the layer-20 output is read at
  prompt_last   last prompt token
  q_last        last token of the question text
  subj_last     last token of the subject's name in the prompt (zeros if not found; see subj_found)
  canvas        all 32 answer positions
for i in SAVE_STEPS. Because Dream attends bidirectionally, prompt positions change from step to step.
Per step and canvas position the top-1 log-prob, the top1-top2 logit margin and whether the slot is still masked are
saved for ALL 32 steps (conf_lp / conf_mg / conf_mask, [N, 32 steps, 32 slots]).

Outputs in --out (row order = rows file order):
  config.json  meta.jsonl (sample_id, version, relation, answer, prompt_len, subj_found, ...)
  hid_prompt_last.npy / hid_q_last.npy / hid_subj_last.npy   [N, S, 3584] float16
  hid_canvas.npy                                             [N, S, 32, 3584] float16   (memmap on disk)
  conf_lp.npy conf_mg.npy [N, 32, 32] float16, conf_mask.npy [N, 32, 32] bool, tokens.npy [N, 32] int32
  done.npy     bool [N]; the run is resumable (rows with done=True are skipped)

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 sae_analysis/dream_steps_extract.py --out sae_analysis/features_dream_steps [--limit-facts 50]
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
from run_inference import build_prompt, extract_answer  # noqa: E402

MODEL = "Dream-org/Dream-v0-Base-7B"
REVISION = "6572adb5535263e4d1a337b56942ba48b6dee2a9"
QWEN = "Qwen/Qwen2.5-7B"
QWEN_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
LAYER = 20
CANVAS, STEPS, BATCH = 32, 32, 64
SAVE_STEPS = [0, 8, 16, 24, 31]
D = 3584
FINAL = "analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl"
RAW = "data/popqa/1. popqa_raw.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=FINAL)
    ap.add_argument("--limit-facts", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    if args.limit_facts:
        keep = list(dict.fromkeys(r["sample_id"] for r in rows))[: args.limit_facts]
        ks = set(keep)
        rows = [r for r in rows if r["sample_id"] in ks]
    n, S = len(rows), len(SAVE_STEPS)
    subj = {}
    for l in open(RAW, encoding="utf-8"):
        x = json.loads(l)
        subj[str(x["id"])] = [m for m in [x["subj"]] + json.loads(x["s_aliases"] or "[]") if m]

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    tokq = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # token positions per row (Qwen fast tokenizer for offsets; same vocabulary as Dream's)
    info = []
    for i, r in enumerate(rows):
        prompt = build_prompt(r["question"])
        enc = tokq(prompt, return_offsets_mapping=True)
        ids, off = enc["input_ids"], enc["offset_mapping"]
        if i < 50:
            assert tok(prompt)["input_ids"] == ids, "Dream and Qwen tokenizers disagree"
        q0 = prompt.find(r["question"])
        q_end = q0 + len(r["question"])
        ql = max(k for k, (a, b) in enumerate(off) if a < q_end and b > q0)
        low, s_end = prompt.lower(), None
        for nm in sorted(subj.get(r["sample_id"], []), key=len, reverse=True):
            j = low.find(nm.lower())
            if j >= 0:
                s_end = max(k for k, (a, b) in enumerate(off) if a < j + len(nm) and b > j)
                break
        info.append((ids, ql, s_end))

    cfg = dict(model=MODEL, revision=REVISION, layer=LAYER, canvas=CANVAS, steps=STEPS, alg="topk_margin", temperature=0.0,
               batch=args.batch_size, save_steps=SAVE_STEPS, rows=args.rows, n_rows=n, padding="none (equal-length batches)")
    json.dump(cfg, open(f"{args.out}/config.json", "w"), indent=1)

    def mm(name, shape, dtype):
        p = f"{args.out}/{name}.npy"
        mode = "r+" if os.path.exists(p) else "w+"
        return np.lib.format.open_memmap(p, mode=mode, dtype=dtype, shape=shape)

    hp, hq, hs = (mm(f"hid_{k}", (n, S, D), np.float16) for k in ("prompt_last", "q_last", "subj_last"))
    hc = mm("hid_canvas", (n, S, CANVAS, D), np.float16)
    clp, cmg = (mm(k, (n, STEPS, CANVAS), np.float16) for k in ("conf_lp", "conf_mg"))
    cmk = mm("conf_mask", (n, STEPS, CANVAS), bool)
    toks = mm("tokens", (n, CANVAS), np.int32)
    done = np.load(f"{args.out}/done.npy") if os.path.exists(f"{args.out}/done.npy") else np.zeros(n, bool)
    found = np.array([s is not None for _, _, s in info])
    answers = {}
    if os.path.exists(f"{args.out}/answers.jsonl"):
        for l in open(f"{args.out}/answers.jsonl", encoding="utf-8"):
            d = json.loads(l)
            answers[d["i"]] = d["answer"]

    model = AutoModel.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    mask_id = tok.mask_token_id
    st = {"step": -1, "buf": None, "p": 0, "ql": None, "se": None}
    save_pos = {s: k for k, s in enumerate(SAVE_STEPS)}

    def layer_hook(module, a, output):
        st["step"] += 1
        k = save_pos.get(st["step"])
        if k is None:
            return
        h = output[0] if isinstance(output, tuple) else output
        b = st["buf"]
        p = st["p"]
        b["pl"][:, k] = h[:, p - 1].to(torch.float16)
        ar = torch.arange(h.shape[0], device=h.device)
        b["ql"][:, k] = h[ar, st["ql"]].to(torch.float16)
        b["se"][:, k] = h[ar, st["se"].clamp(min=0)].to(torch.float16)
        b["cv"][:, k] = h[:, p:p + CANVAS].to(torch.float16)

    model.model.layers[LAYER].register_forward_hook(layer_hook)

    by_len = collections.defaultdict(list)
    for i, (ids, _, _) in enumerate(info):
        if not done[i]:
            by_len[len(ids)].append(i)
    todo = sum(len(v) for v in by_len.values())
    print(f"rows {n}, to do {todo}", flush=True)
    af = open(f"{args.out}/answers.jsonl", "a", encoding="utf-8")
    count = 0
    for length, idxs in sorted(by_len.items()):
        for b0 in range(0, len(idxs), args.batch_size):
            ix = idxs[b0:b0 + args.batch_size]
            nb = len(ix)
            inp = torch.tensor([info[i][0] for i in ix], device="cuda")
            lp = torch.zeros(nb, STEPS, CANVAS, device="cuda")
            mg = torch.zeros_like(lp)
            mk = torch.zeros(nb, STEPS, CANVAS, dtype=torch.bool, device="cuda")
            st.update(step=-1, p=length,
                      ql=torch.tensor([info[i][1] for i in ix], device="cuda"),
                      se=torch.tensor([info[i][2] if info[i][2] is not None else 0 for i in ix], device="cuda"),
                      buf=dict(pl=torch.zeros(nb, S, D, dtype=torch.float16, device="cuda"),
                               ql=torch.zeros(nb, S, D, dtype=torch.float16, device="cuda"),
                               se=torch.zeros(nb, S, D, dtype=torch.float16, device="cuda"),
                               cv=torch.zeros(nb, S, CANVAS, D, dtype=torch.float16, device="cuda")))

            def lhook(step, x, logits):
                z = logits[:, -CANVAS:].float()
                lse = torch.logsumexp(z, -1)
                t2 = z.topk(2, -1).values
                lp[:, step] = t2[..., 0] - lse
                mg[:, step] = t2[..., 0] - t2[..., 1]
                mk[:, step] = x[:, -CANVAS:] == mask_id
                return logits

            with torch.inference_mode():
                out = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS,
                                               temperature=0.0, alg="topk_margin", alg_temp=0.0, mask_token_id=mask_id,
                                               pad_token_id=tok.pad_token_id or tok.eos_token_id, generation_logits_hook_func=lhook)
            assert st["step"] + 1 == STEPS, f"expected {STEPS} forwards, got {st['step'] + 1}"
            order = np.array(ix)
            hp[order] = st["buf"]["pl"].cpu().numpy()
            hq[order] = st["buf"]["ql"].cpu().numpy()
            hs[order] = st["buf"]["se"].cpu().numpy() * found[order][:, None, None]
            hc[order] = st["buf"]["cv"].cpu().numpy()
            clp[order] = lp.cpu().numpy().astype(np.float16)
            cmg[order] = mg.cpu().numpy().astype(np.float16)
            cmk[order] = mk.cpu().numpy()
            toks[order] = out[:, -CANVAS:].cpu().numpy().astype(np.int32)
            for j, i in enumerate(ix):
                ans = extract_answer(tok.decode(out[j][length:], skip_special_tokens=True))
                af.write(json.dumps({"i": i, "sample_id": rows[i]["sample_id"], "version": rows[i]["version"], "answer": ans}, ensure_ascii=False) + "\n")
            af.flush()
            done[order] = True
            count += nb
            if (count // nb) % 20 == 0:
                for m in (hp, hq, hs, hc, clp, cmg, cmk, toks):
                    m.flush()
                np.save(f"{args.out}/done.npy", done)
                print(f"{count}/{todo}", flush=True)
    for m in (hp, hq, hs, hc, clp, cmg, cmk, toks):
        m.flush()
    np.save(f"{args.out}/done.npy", done)
    with open(f"{args.out}/meta.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"i": i, "sample_id": r["sample_id"], "version": r["version"], "relation": r["relation"],
                                "prompt_len": len(info[i][0]), "q_last": info[i][1], "subj_found": bool(found[i])}, ensure_ascii=False) + "\n")
    print("done", n, "subject found:", found.mean(), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
