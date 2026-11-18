"""Prompt-end hidden state at EVERY layer, at denoising step 0 (before any answer slot is filled).  Dream-v0-Base-7B.
Step 0 of generation = one forward of  prompt + 32 mask tokens  (the canvas), so this is that single forward, with the same batching as the stored run
(rows of equal prompt length, batches of 64, rows in file order).  The output of decoder layer L (L = 0..27) at the last prompt token is saved.
Check: layer 20 must match the stored hid_prompt_last[:, 0].
Output: data/1_raw/dream_hidden_layers/hid_prompt_last_layers.npy  [N, 28, 3584] float16 (about 11 GB, not in git).
Usage (repo root): CUDA_VISIBLE_DEVICES=2 python3 analysis/7_sae_vs_hidden/extract_layers.py"""
import collections, json, os, sys
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "data/1_raw"); from run_inference import build_prompt
MODEL = "Dream-org/Dream-v0-Base-7B"; REV = "6572adb5535263e4d1a337b56942ba48b6dee2a9"; CANVAS, BS = 32, 64
OUT = "data/1_raw/dream_hidden_layers"; os.makedirs(OUT, exist_ok=True)
rows = [json.loads(l) for l in open("data/1_raw/qwen_raw.jsonl", encoding="utf-8")]        # the row order of the stored run
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
ids = [tok(build_prompt(r["question"]))["input_ids"] for r in rows]
model = AutoModel.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
NL = len(model.model.layers); n = len(rows)
H = np.lib.format.open_memmap(f"{OUT}/hid_prompt_last_layers.npy", mode="w+", dtype=np.float16, shape=(n, NL, 3584))
st = {}
def hook(L):
    def f(m, a, out):
        h = out[0] if isinstance(out, tuple) else out
        st["buf"][:, L] = h[:, st["p"] - 1].to(torch.float16)
    return f
for L, layer in enumerate(model.model.layers): layer.register_forward_hook(hook(L))
by_len = collections.defaultdict(list)
for i, x in enumerate(ids): by_len[len(x)].append(i)
done = 0
with torch.inference_mode():
    for length, ix_all in sorted(by_len.items()):
        for b0 in range(0, len(ix_all), BS):
            ix = ix_all[b0:b0 + BS]
            inp = torch.tensor([ids[i] + [tok.mask_token_id] * CANVAS for i in ix], device="cuda")
            st.update(p=length, buf=torch.zeros(len(ix), NL, 3584, dtype=torch.float16, device="cuda"))
            model(inp, "full", None)                       # exactly the call diffusion_generate makes when nothing is padded
            H[np.array(ix)] = st["buf"].cpu().numpy(); done += len(ix)
            if done % 6400 < len(ix): print(f"{done}/{n}", flush=True)
H.flush()
S = np.load("data/1_raw/dream_hidden_steps/hid_prompt_last.npy", mmap_mode="r")
chk = np.random.default_rng(0).choice(n, 2000, replace=False); chk.sort()
a = np.asarray(H[chk, 20], dtype=np.float32); b = np.asarray(S[chk, 0], dtype=np.float32)
cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
print(f"check layer 20 vs stored step-0 state (2000 rows): mean cosine {cos.mean():.5f}, min {cos.min():.5f}, max abs diff {np.abs(a - b).max():.4f}")
print("done", n, "rows,", NL, "layers")
