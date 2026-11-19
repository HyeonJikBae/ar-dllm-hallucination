"""E14 디버그 3: 원본 _sample 안의 sample_tokens 출력과 topk 선택을 감싸서 기록, 직접 구현과 step 별 비교."""
import sys, importlib, torch, torch.nn.functional as F
sys.path.insert(0, "followup_mk")
from common import *
import dloop
S = set_S(); V = [i for _, i in S][::5][:80]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in V}
L, ix = next((L, ix) for L, ix in by_length(V, lambda i: ids[i]) if len(ix) >= 8)
inp = torch.tensor([ids[i] for i in ix], device="cuda"); B = len(ix)
gu = sys.modules[type(m.model).__module__.replace("modeling_dream", "generation_utils")]
REC = {"st": [], "tk": []}; _st = gu.sample_tokens; _tk = torch.topk
def st(*a, **kw): r = _st(*a, **kw); REC["st"].append((r[0].clone(), r[1].clone())); return r
def tk(t, k_, *a, **kw):
    r = _tk(t, k_, *a, **kw)
    if t.dim() == 2 and t.shape[0] == B: REC["tk"].append((t.clone(), r[1].clone()))
    return r
gu.sample_tokens = st; gu.torch.topk = tk
with torch.inference_mode():
    m.fns = {}; m.step = -1
    m.model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0, mask_token_id=m.mask_id, pad_token_id=m.tok.pad_token_id or m.tok.eos_token_id)
gu.sample_tokens = _st; torch.topk = _tk
print("기록 수", len(REC["st"]), len(REC["tk"]), flush=True)
with torch.inference_mode():
    x = F.pad(inp, (0, CANVAS), value=m.mask_id); ts = torch.linspace(1, 1e-3, STEPS + 1, device="cuda")
    for i in range(4):
        mi = x == m.mask_id; lg = m.model(x, "full", None).logits; lg = torch.cat([lg[:, :1], lg[:, :-1]], 1)
        conf, x0 = dloop._scores(lg[mi], "topk_margin")
        oc, ox = REC["st"][i]; ofc, oti = REC["tk"][i]
        fc = torch.full_like(x, -torch.inf, dtype=lg.dtype); fc[mi] = conf
        nmt = mi.sum() / B; k_ = int(nmt * (1 - ts[i + 1] / ts[i]))
        _, ti = torch.topk(fc, k_)
        print(f"step {i}: conf 같음 {torch.equal(conf, oc)} (dtype {conf.dtype} 대 {oc.dtype}), x0 같음 {torch.equal(x0, ox)}, full_conf 같음 {torch.equal(fc, ofc)} (dtype {ofc.dtype}), k {k_} 대 {oti.shape[1]}, 선택 같음 {torch.equal(ti, oti)}", flush=True)
        if not torch.equal(conf, oc):
            d = (conf.float() - oc.float()).abs(); print("   conf 최대차", float(d.max()), "다른 수", int((d > 0).sum()), flush=True)
        x_ = torch.zeros_like(x) + m.mask_id; x_[mi] = ox; ri = torch.arange(B, device="cuda").unsqueeze(1).expand_as(oti); x[ri, oti] = x_[ri, oti]   # 원본 선택으로 진행
print("DONE")
