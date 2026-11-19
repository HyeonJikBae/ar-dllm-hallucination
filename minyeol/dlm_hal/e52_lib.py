"""E52 공용: Dream 주의 패치 + 직접 짠 디코딩 고리 (dloop.py 의 orig 모드 복사) + 헤드 직접 로짓 기여 (DLA).

주의 패치: 각 층 self_attn.forward 를 원본 DreamSdpaAttention.forward 와 같은 계산으로 바꾸고, 다음을 덧붙인다.
  - 프롬프트 끝 질의 (L-1) 의 주의 확률을 eager 로 따로 계산해 기록 (ST.cap_probs).  기록은 출력에 영향 없음.
  - 막기 (ST.knock[층] = (막을 키 위치 [B, T] bool, 덮어쓸 행 [B] bool)): 덮어쓸 행의 L-1 질의 출력만 eager 로 다시 계산 (막을 키는 -inf) 해 SDPA 출력에 덮어씀.
    덮어쓸 행이 없으면 SDPA 출력 그대로 -> 빈 막기는 base 와 비트 단위로 같다.
  - 헤드 출력 (o_proj 전, L-1) 기록 (ST.cap_z).
"""
import math, sys
import torch, torch.nn.functional as F
from common import CANVAS, STEPS, extract_answer


class _State:
    active = False; L = None; knock = None; cap_probs = False; probs = None; cap_z = False; z = None
ST = _State()


def _make_fwd(attn, li, arpe, rkv):
    H, d, KV = attn.num_heads, attn.head_dim, attn.num_key_value_heads; G = H // KV; HS = attn.hidden_size; sq = math.sqrt(d)

    def fwd(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False,
            cache_position=None, position_embeddings=None, **kw):
        assert not output_attentions
        bsz, q_len, _ = hidden_states.size()
        query_states = attn.q_proj(hidden_states); key_states = attn.k_proj(hidden_states); value_states = attn.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, H, d).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, KV, d).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, KV, d).transpose(1, 2)
        if position_embeddings is None: cos, sin = attn.rotary_emb(value_states, position_ids)
        else: cos, sin = position_embeddings
        query_states, key_states = arpe(query_states, key_states, cos, sin)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, attn.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
        key_states = rkv(key_states, G); value_states = rkv(value_states, G)
        if query_states.device.type == "cuda" and attention_mask is not None:   # 원본과 같게 ("full" 문자열도 None 이 아님)
            query_states = query_states.contiguous(); key_states = key_states.contiguous(); value_states = value_states.contiguous()
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states,
                                                     attn_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                                                     dropout_p=attn.attention_dropout if attn.training else 0.0, is_causal=False)
        if ST.active:
            P = ST.L - 1; kn = ST.knock.get(li) if ST.knock else None
            do_k = kn is not None and bool(kn[1].any())
            if ST.cap_probs or do_k:
                sc = torch.einsum("bhd,bhtd->bht", query_states[:, :, P].float(), key_states.float()) / sq
                if ST.cap_probs: ST.probs[li] = torch.softmax(sc, -1)
                if do_k:
                    km, rr = kn
                    pk = torch.softmax(sc.masked_fill(km[:, None, :], float("-inf")), -1)
                    o = torch.einsum("bht,bhtd->bhd", pk, value_states.float()).to(attn_output.dtype)
                    attn_output[rr, :, P] = o[rr]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, HS)
        if ST.active and ST.cap_z: ST.z[li] = attn_output[:, ST.L - 1].clone()
        attn_output = attn.o_proj(attn_output)
        return attn_output, None, past_key_value
    return fwd


def patch(m):
    mod = sys.modules[type(m.model.model.layers[0].self_attn).__module__]
    for li, layer in enumerate(m.model.model.layers):
        layer.self_attn.forward = _make_fwd(layer.self_attn, li, mod.apply_rotary_pos_emb, mod.repeat_kv)
    m._fin = {}
    def pre(module, args): m._fin["h"] = args[0]
    m.model.model.norm.register_forward_pre_hook(pre)


_GU = None
def _scores(mask_logits):
    global _GU
    if _GU is None:
        _GU = next(v for k_, v in sys.modules.items() if k_.endswith("generation_utils") and "Dream" in k_ and hasattr(v, "sample_tokens"))
    return _GU.sample_tokens(mask_logits, temperature=0.0, top_p=1.0, top_k=50, margin_confidence=True, neg_entropy=False)


GROUPS = {"all": list(range(28)), "0_9": list(range(10)), "10_19": list(range(10, 20)), "20_27": list(range(20, 28))}


@torch.inference_mode()
def generate(m, id_lists, gold, fns=None, knock=None, layers="all", ko_from=2, cap=False, seed=0, eps=1e-3):
    """dloop.generate (mode orig, topk_margin, 32/32) 와 같은 고리.
    knock: None | 'committed' (프롬프트 끝 -> 확정 캔버스 칸) | 'random' (같은 수의 아직 마스크인 캔버스 칸, 칸 0 제외) | 'empty' (빈 집합) | 'empty_eager' (빈 집합인데 eager 로 덮어씀; 수치 잡음 바닥)
    막기는 step >= ko_from 이고 칸 0 이 아직 마스크인 행에만, 층 묶음 layers 에서.  (eager 다시 계산은 그 행의 모든 층; empty_eager 가 같은 경로의 기준)
    cap: 매 step 프롬프트 끝 주의 기록 -> 확정 칸으로 가는 주의 질량 [B, STEPS, 28, 28].
    반환 dict: answers, rank [B, STEPS] (칸 0 금 첫 토큰 순위, 칸 0 확정 뒤 -2), s0_last (칸 0 이 마스크였던 마지막 step), s0_tok (칸 0 최종 토큰), nk [B, STEPS] (막은 키 수), att_comm."""
    m.fns = dict(fns or {}); m.step = -1
    inp = torch.tensor(id_lists, device="cuda"); B, L = inp.shape; mid_ = m.mask_id; T = L + CANVAS
    x = F.pad(inp, (0, CANVAS), value=mid_)
    ts = torch.linspace(1, eps, STEPS + 1, device=x.device)
    gt = torch.tensor(gold, device="cuda"); rows_ = torch.arange(B, device=x.device)
    rank = torch.full((B, STEPS), -2, dtype=torch.int32, device="cuda"); nk = torch.zeros(B, STEPS, dtype=torch.int16, device="cuda")
    s0m = torch.zeros(B, STEPS, dtype=torch.bool, device="cuda")
    att_comm = torch.zeros(B, STEPS, 28, 28, dtype=torch.float16, device="cuda") if cap else None
    gen = torch.Generator(device="cuda"); nl = len(m.model.model.layers)
    ST.active = True; ST.L = L; ST.cap_probs = cap; ST.probs = {}; ST.cap_z = False
    try:
        for i in range(STEPS):
            mask_index = (x == mid_); m0 = mask_index[:, L]; s0m[:, i] = m0
            ST.knock = {}
            if knock is not None and i >= ko_from:
                comm = ~mask_index[:, L:]
                if knock == "committed": sel = comm.clone()
                elif knock == "random":
                    elig = mask_index[:, L:].clone(); elig[:, 0] = False
                    gen.manual_seed(int(seed) * 1000 + i)
                    r = torch.rand(B, CANVAS, generator=gen, device="cuda"); r[~elig] = 2.0
                    rk_ = r.argsort(1).argsort(1); sel = (rk_ < comm.sum(1, keepdim=True)) & elig
                else: sel = torch.zeros_like(comm)
                sel &= m0[:, None]
                km = torch.zeros(B, T, dtype=torch.bool, device="cuda"); km[:, L:] = sel; k0 = torch.zeros_like(km)
                # 막기 조건과 empty_eager 는 칸 0 이 마스크인 행의 L-1 출력을 모든 층에서 eager 로 다시 계산 (막을 키는 묶음 층에서만 -inf).
                # -> 막기 대 empty_eager 의 차이는 순수하게 막기 효과 (같은 계산 경로).  'empty' 는 아무것도 덮어쓰지 않음 (base 와 비트 단위 같음).
                rr = torch.zeros_like(m0) if knock == "empty" else m0.clone()
                nk[:, i] = sel.sum(1).to(torch.int16)
                for li in range(nl): ST.knock[li] = (km if li in GROUPS[layers] else k0, rr)
            logits = m.model(x, "full", None).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            lf = logits[:, L].float(); gv = lf.gather(1, gt[:, None]); rank[:, i] = torch.where(m0, (lf > gv).sum(1).int(), rank[:, i])
            if cap:
                comm = (~mask_index[:, L:]).float()
                for li in range(nl): att_comm[:, i, li] = (ST.probs[li][:, :, L:] * comm[:, None, :]).sum(-1).half()
            mask_logits = logits[mask_index]
            t = ts[i]; s = ts[i + 1]
            confidence, x0 = _scores(mask_logits)
            full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            ntt = int(num_mask_token * (1 - s / t)) if i < STEPS - 1 else int(num_mask_token)
            if ntt > 0:
                _, transfer_index = torch.topk(full_confidence, ntt)
                x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mid_; x_[mask_index] = x0.clone()
                ri = rows_.unsqueeze(1).expand_as(transfer_index); x[ri, transfer_index] = x_[ri, transfer_index]
    finally:
        ST.active = False; ST.knock = None; ST.probs = None; m.fns = {}
    out = []
    for j in range(B):
        g_ = x[j, L:].tolist(); g_ = g_[: next((q for q, t_ in enumerate(g_) if t_ in m.SP), len(g_))]; out.append(extract_answer(m.tok.decode(g_, skip_special_tokens=True)))
    s0m_ = s0m.cpu(); last = [int(torch.where(s0m_[j])[0].max()) if s0m_[j].any() else -1 for j in range(B)]
    tok0 = x[:, L].tolist()   # 칸 0 에 확정된 토큰
    res = {"answers": out, "rank": rank.cpu().numpy(), "s0_last": last, "s0_tok": tok0, "nk": nk.cpu().numpy(), "canvas": x[:, L:].cpu().numpy()}
    if cap: res["att_comm"] = att_comm.float().cpu().numpy()
    return res


@torch.inference_mode()
def step0(m, id_lists, gold, comp, subj, fns=None, qlast=None):
    """step 0 (캔버스 전부 마스크) 한 번 forward.  헤드별 DLA (금, 경쟁), 프롬프트 끝 주의 (주어 끝, 캔버스 합), 칸 0 순위.
    DLA_{l,h}(t) = s * (W_U[t] * w_norm) . (W_O[:, h] z_{l,h}),  s = 그 행의 실제 최종 RMSNorm 배율 (선형화).
    gold, comp: 토큰 id 목록 (comp < 0 이면 0 으로 두고 결과 무시).  subj: 주어 끝 위치 (없으면 -1)."""
    m.fns = dict(fns or {}); m.step = -1
    inp = torch.tensor(id_lists, device="cuda"); B, L = inp.shape; mid_ = m.mask_id
    x = F.pad(inp, (0, CANVAS), value=mid_); nl = len(m.model.model.layers)
    ST.active = True; ST.L = L; ST.cap_probs = True; ST.probs = {}; ST.cap_z = True; ST.z = {}; ST.knock = {}
    try:
        logits = m.model(x, "full", None).logits
    finally:
        ST.active = False; m.fns = {}
    lf = logits[:, L - 1].float()                      # 칸 0 의 로짓 (한 칸 밀기 전 L-1)
    gt = torch.tensor(gold, device="cuda"); ct = torch.tensor([max(c, 0) for c in comp], device="cuda")
    h = m._fin["h"][:, L - 1].float(); eps_ = m.model.model.norm.variance_epsilon
    s = torch.rsqrt(h.pow(2).mean(-1) + eps_)          # [B]
    wn = m.model.model.norm.weight.float(); WU = m.model.lm_head.weight
    ug = WU[gt].float() * wn; uc = WU[ct].float() * wn
    dg = torch.zeros(B, nl, 28, device="cuda"); dc = torch.zeros(B, nl, 28, device="cuda")
    for li in range(nl):
        Wo = m.model.model.layers[li].self_attn.o_proj.weight.float()   # [out, in]
        z = ST.z[li].float().view(B, 28, -1)
        dg[:, li] = (ug @ Wo).view(B, 28, -1).mul(z).sum(-1) * s[:, None]
        dc[:, li] = (uc @ Wo).view(B, 28, -1).mul(z).sum(-1) * s[:, None]
    P = torch.stack([ST.probs[li] for li in range(nl)], 1)   # [B, 28, 28, T]
    sj = torch.tensor([max(v, 0) for v in subj], device="cuda")
    att_subj = P.gather(3, sj.view(B, 1, 1, 1).expand(B, nl, 28, 1)).squeeze(-1)
    att_canvas = P[..., L:].sum(-1)
    qt = torch.tensor([max(v, 0) for v in (qlast or [0] * B)], device="cuda")
    att_ql = P.gather(3, qt.view(B, 1, 1, 1).expand(B, nl, 28, 1)).squeeze(-1); att_p0 = P[..., 0]
    gv = lf.gather(1, gt[:, None]); cv = lf.gather(1, ct[:, None])
    res = {"dla_g": dg.cpu().numpy(), "dla_c": dc.cpu().numpy(), "scale": s.cpu().numpy(), "att_subj": att_subj.cpu().numpy(), "att_canvas": att_canvas.cpu().numpy(), "att_ql": att_ql.cpu().numpy(), "att_p0": att_p0.cpu().numpy(),
           "logit_diff": (gv - cv)[:, 0].cpu().numpy(), "lin_diff": (s * ((ug - uc) * h).sum(-1)).cpu().numpy(), "rank0": (lf > gv).sum(1).cpu().numpy(),
           "top1": lf.argmax(1).cpu().numpy()}
    ST.z = None; ST.probs = None
    return res
