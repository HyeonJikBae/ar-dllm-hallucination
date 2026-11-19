"""E14 디버그 2: step 1 에서 sort 기반 margin 과 topk 기반 margin, 선택 위치 비교."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "followup_mk")
from common import *
import dloop
S = set_S(); V = [i for _, i in S][::5][:80]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in V}
L, ix = next((L, ix) for L, ix in by_length(V, lambda i: ids[i]) if len(ix) >= 8)
inp = torch.tensor([ids[i] for i in ix], device="cuda"); B = len(ix)
with torch.inference_mode():
    x = F.pad(inp, (0, CANVAS), value=m.mask_id); ts = torch.linspace(1, 1e-3, STEPS + 1, device="cuda")
    for i in range(3):
        mi = x == m.mask_id; lg = m.model(x, "full", None).logits; lg = torch.cat([lg[:, :1], lg[:, :-1]], 1); ml = lg[mi]
        probs = torch.softmax(ml, -1); c0, x0 = probs.max(-1)
        sp, _ = torch.sort(probs, -1, descending=True); cs = sp[:, 0] - sp[:, 1]
        ct, _ = dloop._scores(ml, "topk_margin")
        tk = torch.topk(probs, 2, -1).values
        print(f"step {i}: sort top2 대 topk top2 같음 {torch.equal(sp[:, :2], tk)}, margin 같음 {torch.equal(cs, ct)}, 최대차 {float((cs.float() - ct.float()).abs().max())}, sort top1==max {torch.equal(sp[:, 0], c0)}", flush=True)
        if not torch.equal(sp[:, :2], tk):
            bad = (sp[:, :2] != tk).any(1).nonzero().flatten()[:3]
            for b in bad.tolist(): print("  예", sp[b, :3].tolist(), tk[b].tolist(), flush=True)
        fc = torch.full_like(x, -torch.inf, dtype=lg.dtype); fc[mi] = cs
        nmt = mi.sum() / B; k_ = int(nmt * (1 - ts[i + 1] / ts[i]))
        _, ti = torch.topk(fc, k_); x_ = torch.zeros_like(x) + m.mask_id; x_[mi] = x0; ri = torch.arange(B, device="cuda").unsqueeze(1).expand_as(ti); x[ri, ti] = x_[ri, ti]
print("DONE")
