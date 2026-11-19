"""E14 재현 디버그: 한 배치에서 원본을 두 번, dloop 를 두 번 돌려 step 별 x 를 비교."""
import collections, sys
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import dloop
S = set_S(); V = [i for _, i in S][::5][:80]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in V}
L, ix = next((L, ix) for L, ix in by_length(V, lambda i: ids[i]) if len(ix) >= 8)
inp = torch.tensor([ids[i] for i in ix], device="cuda"); print("배치", len(ix), "길이", L, flush=True)
def orig():
    H = []; LG = []
    def th(i, x, logits):
        if i is not None: H.append(x[:, L:].clone().cpu())
        return x
    def lh(i, x, logits): LG.append(logits[:, L:].float().cpu()); return logits
    with torch.inference_mode():
        m.fns = {}; m.step = -1
        o = m.model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                       mask_token_id=m.mask_id, pad_token_id=m.tok.pad_token_id or m.tok.eos_token_id, generation_tokens_hook_func=th, generation_logits_hook_func=lh)
    return H, LG
def mine():
    import torch.nn.functional as F
    H = []; LG = []
    with torch.inference_mode():
        m.fns = {}; m.step = -1; x = F.pad(inp, (0, CANVAS), value=m.mask_id); ts = torch.linspace(1, 1e-3, STEPS + 1, device="cuda"); B = len(ix)
        for i in range(STEPS):
            mi = x == m.mask_id; lg = m.model(x, "full", None).logits; lg = torch.cat([lg[:, :1], lg[:, :-1]], 1); LG.append(lg[:, L:].float().cpu())
            conf, x0 = dloop._scores(lg[mi], "topk_margin"); fc = torch.full_like(x, -torch.inf, dtype=lg.dtype); fc[mi] = conf
            nmt = mi.sum() / B; k_ = int(nmt * (1 - ts[i + 1] / ts[i])) if i < STEPS - 1 else int(nmt)
            if k_ > 0:
                _, ti = torch.topk(fc, k_); x_ = torch.zeros_like(x) + m.mask_id; x_[mi] = x0; ri = torch.arange(B, device="cuda").unsqueeze(1).expand_as(ti); x[ri, ti] = x_[ri, ti]
            H.append(x[:, L:].clone().cpu())
    return H, LG
o1, l1 = orig(); o2, l2 = orig(); d1, k1 = mine(); d2, k2 = mine()
def first_div(A, B_):
    for s, (a, b) in enumerate(zip(A, B_)):
        if not torch.equal(a, b): return s, int((a != b).any(1).sum())
    return None
print("원본1 대 원본2:", first_div(o1, o2), " logits step0 최대차", float((l1[0] - l2[0]).abs().max()), flush=True)
print("직접1 대 직접2:", first_div(d1, d2), flush=True)
print("원본1 대 직접1:", first_div(o1, d1), " logits step0 최대차", float((l1[0] - k1[0]).abs().max()), " dtype 원본 logits?", flush=True)
s = first_div(o1, d1)
if s:
    s = s[0]; print("step", s, "logits 최대차", float((l1[s] - k1[s]).abs().max()))
print("DONE")
