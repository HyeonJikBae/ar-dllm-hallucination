"""Prompt-end hidden state at EVERY layer and at denoising steps 0 / 4 / 8 / 16 / 31.  Dream-v0-Base-7B.
States after step 0 depend on the answer slots already filled, so the generation must be the stored one: the whole data set is re-generated with the
stored batching (rows of equal prompt length, batches of 64, file order), and the final canvas is checked against data/1_raw/dream_hidden_steps/tokens.npy.
Only the rows needed for the analysis are saved: every row of facts 1/5..4/5, every row of 5/5 facts, 5,000 random rows of 0/5 facts (seed 0).
Output: data/1_raw/dream_hidden_layers/hid_prompt_last_layers_steps.npy  [n_saved, 5 steps, 28 layers, 3584] float16, rows listed in rows_layers_steps.npy,
        same_canvas_layers_steps.npy (bool per saved row: final canvas identical to the stored run).
Usage (repo root): CUDA_VISIBLE_DEVICES=2 python3 analysis/7_sae_vs_hidden/extract_layers_steps.py"""
import collections, json, os, sys
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "data/1_raw"); from run_inference import build_prompt
MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; CANVAS, STEPS, BS = 32, 32, 64
SAVE_STEPS = [0, 4, 8, 16, 31]; OUT = "data/1_raw/dream_hidden_layers"
rows = [json.loads(l) for l in open("data/1_raw/qwen_raw.jsonl", encoding="utf-8")]            # row order of the stored run
lab = [json.loads(l) for l in open("data/3_classification/full/dream_final_full.jsonl", encoding="utf-8")]
assert all(a["sample_id"] == b["sample_id"] and a["version"] == b["version"] for a, b in zip(rows, lab))
k = np.array([r["k_correct"] for r in lab]); n = len(rows)
z = np.where(k == 0)[0]; keep = np.sort(np.concatenate([np.where(k >= 1)[0], np.random.default_rng(0).choice(z, 5000, replace=False)]))
slot = np.full(n, -1); slot[keep] = np.arange(len(keep)); np.save(f"{OUT}/rows_layers_steps.npy", keep)
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
ids = [tok(build_prompt(r["question"]))["input_ids"] for r in rows]
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
NL = len(model.model.layers); mask_id = tok.mask_token_id
# Memory: Dream's sample_tokens sorts the whole vocabulary for every masked slot at once.  Each slot is handled independently,
# so running it in chunks of slots gives exactly the same tokens with a much smaller peak.  The forward pass itself keeps batch 64.
gu = next(m for name, m in sys.modules.items() if name.endswith("generation_utils") and hasattr(m, "sample_tokens") and "Dream" in name)
_orig = gu.sample_tokens
def _chunked(logits, *a, **kw):
    if logits.shape[0] <= 256: return _orig(logits, *a, **kw)
    outs = [_orig(logits[s:s + 256], *a, **kw) for s in range(0, logits.shape[0], 256)]
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
gu.sample_tokens = _chunked
H = np.lib.format.open_memmap(f"{OUT}/hid_prompt_last_layers_steps.npy", mode="w+", dtype=np.float16, shape=(len(keep), len(SAVE_STEPS), NL, 3584))
stored = np.load("data/1_raw/dream_hidden_steps/tokens.npy", mmap_mode="r"); same = np.zeros(len(keep), bool)
pos = {s: j for j, s in enumerate(SAVE_STEPS)}; st = {"step": -1}
def hook(L):
    def f(m, a, out):
        if L == 0: st["step"] += 1
        j = pos.get(st["step"])
        if j is None: return
        h = out[0] if isinstance(out, tuple) else out
        st["buf"][:, j, L] = h[:, st["p"] - 1].to(torch.float16)
    return f
for L, layer in enumerate(model.model.layers): layer.register_forward_hook(hook(L))
by_len = collections.defaultdict(list)
for i, x in enumerate(ids): by_len[len(x)].append(i)
done = 0
for length, ix_all in sorted(by_len.items()):
    for b0 in range(0, len(ix_all), BS):
        ix = ix_all[b0:b0 + BS]; inp = torch.tensor([ids[i] for i in ix], device="cuda")
        st.update(step=-1, p=length, buf=torch.zeros(len(ix), len(SAVE_STEPS), NL, 3584, dtype=torch.float16, device="cuda"))
        with torch.inference_mode():
            o = model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                         mask_token_id=mask_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        assert st["step"] + 1 == STEPS
        ix = np.array(ix); sel = slot[ix] >= 0
        if sel.any():
            H[slot[ix[sel]]] = st["buf"][torch.from_numpy(sel).cuda()].cpu().numpy()
            same[slot[ix[sel]]] = (o[:, -CANVAS:].cpu().numpy() == stored[ix])[sel].all(1)
        done += len(ix)
        if done % 6400 < len(ix): print(f"{done}/{n}  saved rows identical so far: {same[slot[ix[sel]]].mean() if sel.any() else float('nan'):.3f}", flush=True)
H.flush(); np.save(f"{OUT}/same_canvas_layers_steps.npy", same)
print(f"done. saved {len(keep)} rows; final canvas identical to the stored run: {same.mean():.4f}")
