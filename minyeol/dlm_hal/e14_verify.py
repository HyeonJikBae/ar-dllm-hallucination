"""E14 재현 검증: set_S() 300 행 (set 마다 앞 60) 을 원본 diffusion_generate 와 dloop orig 로 같은 배치에서 돌려 캔버스 전체 비교, E7·E9 base 답과 비교.
또 t2m 판정에 쓰는 '보이는 답 칸의 p(현재 토큰)' 분포를 기록한다 (규칙이 실제로 걸리는지 확인용)."""
import collections, json, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import dloop
S = set_S(); per = collections.defaultdict(list)
for s_, i in S: per[s_].append(i)
V = [i for s_ in per for i in per[s_][:60]]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in V}
e7 = {}
for l in open(f"{OUT}/e7/meta.jsonl"): d = json.loads(l); e7[d["row"]] = d["answer"]
e9 = {}
for l in open(f"{OUT}/results_e9.jsonl"):
    d = json.loads(l)
    if d["cond"] == "base": e9[d["row"]] = d["answer"]
same_canvas = 0; same_ans = 0; n = 0; a7 = []; a9 = []; t0 = time.time()
for L, ix in by_length(V, lambda i: ids[i]):
    inp = torch.tensor([ids[i] for i in ix], device="cuda")
    with torch.inference_mode():
        m.fns = {}; m.step = -1
        o = m.model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="topk_margin", alg_temp=0.0,
                                       mask_token_id=m.mask_id, pad_token_id=m.tok.pad_token_id or m.tok.eos_token_id)
    orig_canvas = o[:, L:].cpu()
    ans, canv, _ = dloop.generate(m, [ids[i] for i in ix], return_canvas=True)
    eq = (orig_canvas == canv).all(1)
    same_canvas += int(eq.sum()); n += len(ix)
    ref = m.generate([ids[i] for i in ix])
    same_ans += sum(a == b for a, b in zip(ans, ref))
    for i, a in zip(ix, ans):
        if i in e7: a7.append(a == e7[i])
        if i in e9: a9.append(a == e9[i])
print(f"행 {n}: 원본과 캔버스 전체 일치 {same_canvas}/{n}, 원본 답 일치 {same_ans}/{n}, E7 답 일치 {sum(a7)}/{len(a7)}, E9 base 답 일치 {sum(a9)}/{len(a9)}  ({time.time() - t0:.0f}s)", flush=True)
# 보이는 답 칸의 p(현재 토큰) 분포: 원본 고리에서 step 마다 기록
P = []
def probe(L, ix):
    m.fns = {}; m.step = -1; inp = torch.tensor([ids[i] for i in ix], device="cuda"); B = len(ix)
    x = torch.nn.functional.pad(inp, (0, CANVAS), value=m.mask_id); ts = torch.linspace(1, 1e-3, STEPS + 1, device="cuda"); SPt = torch.tensor(sorted(m.SP), device="cuda")
    for i_ in range(STEPS):
        mi = x == m.mask_id; lg = m.model(x, "full", None).logits; lg = torch.cat([lg[:, :1], lg[:, :-1]], 1)
        vis = ~mi[:, L:]; tok = x[:, L:]; isp = torch.isin(tok, SPt) & vis
        fs = torch.where(isp.any(1), isp.float().argmax(1), torch.full((B,), CANVAS, device="cuda")); ct = vis & ~isp & (torch.arange(CANVAS, device="cuda")[None] < fs[:, None])
        if ct.any(): P.extend(torch.softmax(lg[:, L:][ct].float(), -1).gather(-1, tok[ct].unsqueeze(-1)).squeeze(-1).tolist())
        conf, x0 = dloop._scores(lg[mi], "topk_margin"); fc = torch.full_like(x, -torch.inf, dtype=lg.dtype); fc[mi] = conf
        nmt = mi.sum() / B; k_ = int(nmt * (1 - ts[i_ + 1] / ts[i_])) if i_ < STEPS - 1 else int(nmt)
        if k_ > 0:
            _, ti = torch.topk(fc, k_); x_ = torch.zeros_like(x) + m.mask_id; x_[mi] = x0; ri = torch.arange(B, device="cuda").unsqueeze(1).expand_as(ti); x[ri, ti] = x_[ri, ti]
with torch.inference_mode():
    for L, ix in list(by_length(V, lambda i: ids[i]))[:6]: probe(L, ix)
P = np.array(P); print(f"보이는 답 내용 칸 p(현재 토큰): n {len(P)}, 백분위 1/5/25/50 = {np.percentile(P, [1, 5, 25, 50]).round(4).tolist()}, <0.3 {100 * (P < .3).mean():.2f}%, <0.5 {100 * (P < .5).mean():.2f}%", flush=True)
json.dump(dict(n=n, canvas_same=same_canvas, ans_same=same_ans, e7_same=[int(sum(a7)), len(a7)], e9_same=[int(sum(a9)), len(a9)], p_vis_pct=np.percentile(P, [1, 5, 25, 50]).tolist(), p_lt03=float((P < .3).mean()), p_lt05=float((P < .5).mean())),
          open(f"{OUT}/e14_verify.json", "w"), indent=1)
print("DONE", flush=True)
