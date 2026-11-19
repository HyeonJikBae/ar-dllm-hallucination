"""E13 (PREREG.md): E7 궤적으로 step 0,1,2,3,8 캔버스를 다시 만들어 한 번씩 forward.  주의 요약, 칸 0 logit lens, 칸 0~7 분포, 금 답 첫 토큰 순위.
재시작 가능.  usage: python followup_mk/e13_capture.py [--pilot] [--gpu N]
칸 j 의 logit 은 위치 L+j-1 의 은닉에서 나온다 (Dream 은 logits 를 한 칸 민다).  그래서 칸 0 의 예측 위치 = 프롬프트 끝.
주의 질의 위치 QP = [프롬프트 끝(L-1, 칸 0 예측), 캔버스 0, 1, 2, 3 (각각 칸 1~4 예측)].
키 구간 SEG = [0 첫 토큰(sink), 1 지시문·나머지 프롬프트, 2 질문 중 주어 밖, 3 주어, 4 캔버스]."""
import json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
PIL = "--pilot" in sys.argv
D7 = f"{OUT}/e7"; D13 = f"{OUT}/e13{'_pilot' if PIL else ''}"; os.makedirs(D13, exist_ok=True)
R7 = np.load(f"{D7}/rows.npy"); w7 = {int(i): j for j, i in enumerate(R7)}
_, Z, Fv = e7_rows(); MIDR = np.where(mid)[0]
if PIL: R = np.concatenate([MIDR[::50][:120], Z[:40], Fv[:40]]).astype(int)
else: R = np.concatenate([MIDR, Z[:1000], Fv[:1000]]).astype(int)
n = len(R); SV = [0, 1, 2, 3, 8]; NL = 28; NH = 28; NQ = 5; NS = 5; KS = 8
if os.path.exists(f"{D13}/rows.npy"): assert (np.load(f"{D13}/rows.npy") == R).all()
np.save(f"{D13}/rows.npy", R); json.dump(dict(steps=SV, qpos=["prompt_end(slot0)", "canvas0", "canvas1", "canvas2", "canvas3"], seg=["sink", "instr", "question_rest", "subject", "canvas"]), open(f"{D13}/config.json", "w"))
mm = lambda name, shape, dt: np.lib.format.open_memmap(f"{D13}/{name}.npy", mode="r+" if os.path.exists(f"{D13}/{name}.npy") else "w+", dtype=dt, shape=shape)
ATT = mm("att", (n, len(SV), NL, NH, NQ, NS), np.float16)
LL_TOP = mm("ll_top1", (n, len(SV), NL), np.int32); LL_PT = mm("ll_pfinal", (n, len(SV), NL), np.float16); LL_ENT = mm("ll_ent", (n, len(SV), NL), np.float16)
LL_JS = mm("ll_js", (n, len(SV), NL), np.float16); LL_GR = mm("ll_grank", (n, len(SV), NL), np.int32)
ENT = mm("ent", (n, len(SV), KS), np.float16); TOPI = mm("top10_id", (n, len(SV), KS, 10), np.int32); TOPP = mm("top10_p", (n, len(SV), KS, 10), np.float16); AM = mm("argmax", (n, len(SV), KS), np.int32)
done = np.load(f"{D13}/done.npy") if os.path.exists(f"{D13}/done.npy") else np.zeros(n, bool)
# ---- 입력 재구성
I7 = np.array([w7[int(i)] for i in R]); IDS7 = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[np.sort(I7)]); MK7 = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[np.sort(I7)])
o_ = np.argsort(np.argsort(I7)); IDS7 = IDS7[o_]; MK7 = MK7[o_]                                   # R 순서로
lastm = np.where(MK7, np.arange(STEPS)[None, :, None], -1).max(1); COMM = np.take_along_axis(IDS7, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
from transformers import AutoTokenizer
_tq = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
def segs(i):
    r = rows[i]; p = build_prompt(r["question"]); enc = _tq(p, return_offsets_mapping=True); off = enc["offset_mapping"]
    q0 = p.find(r["question"]); qe = q0 + len(r["question"]); s = np.ones(len(off), np.int8)
    for t, (a, b) in enumerate(off):
        if a < qe and b > q0: s[t] = 2
    x = POP[r["sample_id"]]; low = p.lower(); found = False
    for nm in sorted([m_ for m_ in [x["subj"]] + json.loads(x["s_aliases"] or "[]") if m_], key=len, reverse=True):
        j = low.find(nm.lower(), q0)                                                           # 질문 안에서만 찾음
        if 0 <= j < qe:
            for t, (a, b) in enumerate(off):
                if a < j + len(nm) and b > j: s[t] = 3
            found = True; break
    s[0] = 0
    return enc["input_ids"], s, found
SEG = {}; meta = {}
for j, i in enumerate(R):
    ids_, s_, f_ = segs(int(i)); SEG[j] = (ids_, s_); meta[j] = dict(row=int(i), L=len(ids_), subj_found=f_, n_subj=int((s_ == 3).sum()))
json.dump([meta[j] for j in range(n)], open(f"{D13}/meta.json", "w"))
def gold_first(i):
    r = rows[i]; al = r["gold_aliases"] if isinstance(r["gold_aliases"], list) else json.loads(r["gold_aliases"] or "[]"); c = set()
    for a in [r["gold_value"]] + al:
        a = (a or "").strip()
        if not a: continue
        for v in (" " + a, a, " " + a[0].upper() + a[1:], " " + a.lower()):
            t = _tq(v, add_special_tokens=False)["input_ids"]
            if t: c.add(t[0])
    return sorted(c)
GOLD = {j: gold_first(int(i)) for j, i in enumerate(R)}
# ---- 모델과 훅
gpu = int(sys.argv[sys.argv.index("--gpu") + 1]) if "--gpu" in sys.argv else 0
torch.cuda.set_device(gpu)
m = Dream(); TOK = m.tok
for j in range(n): assert m.ids(build_prompt(rows[int(R[j])]["question"])) == SEG[j][0], "토큰화 불일치"
rotate_half = sys.modules[type(m.model.model.layers[0].self_attn).__module__].rotate_half
CAP = {"on": False}
def mkpre(l_):
    def pre(mod, args, kw):
        if not CAP["on"]: return
        hs = kw["hidden_states"]; cos, sin = kw["position_embeddings"]; B, N, _ = hs.shape; qp = CAP["qp"]; hd = mod.head_dim
        q = mod.q_proj(hs[:, qp]).view(B, len(qp), mod.num_heads, hd).transpose(1, 2); kk_ = mod.k_proj(hs).view(B, N, mod.num_key_value_heads, hd).transpose(1, 2)
        q = q * cos[:, qp].unsqueeze(1) + rotate_half(q) * sin[:, qp].unsqueeze(1); kk_ = kk_ * cos.unsqueeze(1) + rotate_half(kk_) * sin.unsqueeze(1)
        kk_ = kk_.repeat_interleave(mod.num_key_value_groups, dim=1)
        a = torch.softmax((q.float() @ kk_.float().transpose(-1, -2)) / math.sqrt(hd), -1)            # [B, H, Q, N]
        CAP["att"][:, l_] = (a @ CAP["oh"].unsqueeze(1)).half()                                       # [B, H, Q, S]
        if CAP.get("check") == l_: CAP["chk"] = (a, hs, cos, sin)
    return pre
for l_, layer in enumerate(m.model.model.layers): layer.self_attn.register_forward_pre_hook(mkpre(l_), with_kwargs=True)
norm_f = m.model.model.norm; head = m.model.lm_head
def js(lp, lq):
    p, q = lp.exp(), lq.exp(); mlog = torch.log(0.5 * (p + q) + 1e-30); return 0.5 * ((p * (lp - mlog)).sum(-1) + (q * (lq - mlog)).sum(-1))
@torch.inference_mode()
def run_batch(jj, L, check=False):
    B = len(jj); P = torch.tensor(np.stack([SEG[j][0] for j in jj]), device="cuda")
    oh = torch.zeros(B, L + CANVAS, NS, device="cuda"); sg = torch.tensor(np.stack([SEG[j][1] for j in jj]), device="cuda").long()
    oh[:, :L].scatter_(2, sg[:, :, None], 1.0); oh[:, L:, 4] = 1.0
    CAP.update(on=True, qp=torch.tensor([L - 1, L, L + 1, L + 2, L + 3], device="cuda"), oh=oh)
    gl = [torch.tensor(GOLD[j], device="cuda") for j in jj]; out = {}
    for si, s in enumerate(SV):
        cv = torch.tensor(np.where(MK7[jj, s], m.mask_id, COMM[jj]), device="cuda"); x = torch.cat([P, cv], 1)
        CAP["att"] = torch.zeros(B, NL, NH, NQ, NS, dtype=torch.float16, device="cuda"); hl = torch.zeros(B, NL, 3584, dtype=torch.bfloat16, device="cuda")
        def mk(l_):
            def f(h, step): hl[:, l_] = h[:, L - 1]
            return f
        m.fns = {l_: mk(l_) for l_ in range(NL)}; m.step = -1
        if check and si == 0: CAP["check"] = 10
        lg = m.model(input_ids=x, attention_mask="full", num_logits_to_keep=CANVAS + 1).logits[:, :CANVAS]        # [B, 32, V] 칸 j
        m.fns = {}; CAP["check"] = None
        am = torch.softmax(lg[:, :KS], -1).max(-1).indices
        lf = torch.log_softmax(lg[:, :KS].float(), -1); ent = -(lf.exp() * lf).sum(-1); tp = lf.topk(10, -1)
        lens = torch.log_softmax(head(norm_f(hl)).float(), -1)                                      # [B, NL, V]
        fin = lf[:, 0]; ftop = fin.argmax(-1)
        lt = lens.argmax(-1); lpt = lens.gather(-1, ftop[:, None, None].expand(B, NL, 1))[..., 0].exp(); le = -(lens.exp() * lens).sum(-1); lj = js(lens, fin[:, None, :].expand_as(lens))
        gr = torch.stack([(lens[b] > lens[b][:, gl[b]].max(-1).values[:, None]).sum(-1) for b in range(B)])
        out[si] = dict(att=CAP["att"].float().cpu().numpy(), am=am.cpu().numpy(), ent=ent.cpu().numpy(), ti=tp.indices.cpu().numpy(), tpp=tp.values.exp().cpu().numpy(),
                       lt=lt.cpu().numpy(), lpt=lpt.cpu().numpy(), le=le.cpu().numpy(), lj=lj.cpu().numpy(), gr=gr.cpu().numpy())
        if check and si == 0:
            a, hs, cos, sin = CAP["chk"]; mod = m.model.model.layers[10].self_attn
            _, w, _ = type(mod).__mro__[1].forward(mod, hidden_states=hs, attention_mask=None, position_embeddings=(cos, sin), output_attentions=True)
            d = (w[:, :, CAP["qp"]].float() - a).abs().max().item(); print(f"  주의 점검 (층 10, eager 와 최대 차이): {d:.2e}", flush=True)
            print(f"  마지막 층 lens 와 칸 0 logits 의 top-1 일치: {(lt[:, -1] == ftop).float().mean().item():.4f}, 마지막 층 JS 평균 {lj[:, -1].mean().item():.2e}", flush=True)
    CAP["on"] = False
    return out
groups = {}
for j in range(n):
    if not done[j]: groups.setdefault(meta[j]["L"], []).append(j)
t0 = time.time(); cnt = 0; bs = 16; first = True
for L, js_ in sorted(groups.items()):
    b0 = 0
    while b0 < len(js_):
        jj = np.array(js_[b0:b0 + bs])
        try: out = run_batch(jj, L, check=first)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); bs = max(1, bs // 2); print(f"  OOM -> bs {bs}", flush=True); continue
        first = False
        for si in range(len(SV)):
            o = out[si]; ATT[jj, si] = o["att"]; AM[jj, si] = o["am"]; ENT[jj, si] = o["ent"]; TOPI[jj, si] = o["ti"]; TOPP[jj, si] = o["tpp"]
            LL_TOP[jj, si] = o["lt"]; LL_PT[jj, si] = o["lpt"]; LL_ENT[jj, si] = o["le"]; LL_JS[jj, si] = o["lj"]; LL_GR[jj, si] = o["gr"]
        done[jj] = True; cnt += len(jj); b0 += len(jj)
        if cnt % 320 < len(jj):
            for a_ in (ATT, AM, ENT, TOPI, TOPP, LL_TOP, LL_PT, LL_ENT, LL_JS, LL_GR): a_.flush()
            np.save(f"{D13}/done.npy", done); print(f"  {cnt}/{int((~done).sum()) + cnt} {time.time() - t0:.0f}s", flush=True)
for a_ in (ATT, AM, ENT, TOPI, TOPP, LL_TOP, LL_PT, LL_ENT, LL_JS, LL_GR): a_.flush()
np.save(f"{D13}/done.npy", done)
# 재구성 일치 (칸 0~7 argmax 대 E7 ids)
agree = {f"s{s}": round(float((np.asarray(AM[:, si]) == IDS7[:, s, :KS]).mean()) * 100, 2) for si, s in enumerate(SV)}
agree0 = {f"s{s}": round(float((np.asarray(AM[:, si, 0]) == IDS7[:, s, 0]).mean()) * 100, 2) for si, s in enumerate(SV)}
print("재구성 argmax 일치 (칸 0~7):", agree, " 칸 0:", agree0, flush=True)
json.dump(dict(agree_slot0_7=agree, agree_slot0=agree0, n=n), open(f"{D13}/check.json", "w"))
print("DONE", flush=True)
