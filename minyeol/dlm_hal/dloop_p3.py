"""E14: Dream diffusion_generate (_sample, temperature 0, alg_temp 0) 를 직접 구현.  common.Dream 의 층 훅 (fns) 과 함께 쓴다.
mode:
  "orig"  : 원본과 같은 동작.  옮기는 개수는 원본처럼 배치 평균 마스크 수로 계산 (행마다 마스크 수가 같을 때만 원본과 같음).
  "t2m"   : Remask, Don't Replace (arXiv 2604.18738) 의 LowProb 규칙.  각 step 의 M2T 확정 뒤, 이번 forward 에서 보이던 답 내용 칸의
            p(현재 토큰 | 문맥) < tau 이면 다시 가림 (칸마다 최대 cmax 번, 마지막 step 제외).  마스크 수가 행마다 달라지므로 옮기는 개수는 행별.
  "defer" : 가려진 동안 argmax 가 defer_k 번 이상 바뀐 답 칸 (현재 argmax 가 특수 토큰이 아닌 칸) 의 확정 우선순위를 맨 뒤로.
            (주파수 기반 다시 가리기 arXiv 2611.10486 은 반대로 자주 바뀌는 칸을 먼저 확정한다.  여기서는 사전 지시대로 미룸.)
"""
import torch, torch.nn.functional as F
from common import CANVAS, STEPS, extract_answer


_GU = None
def _scores(mask_logits, alg, m=None):
    """원본 sample_tokens 를 그대로 부른다.  Dream 은 생성 설정을 모델 config 에서 만들어 top_k=50, top_p=1.0 이 들어간다 (확인함)."""
    global _GU
    if _GU is None:
        import sys as _s
        _GU = next(v for k_, v in _s.modules.items() if k_.endswith("generation_utils") and "Dream" in k_ and hasattr(v, "sample_tokens"))
    if alg not in ("topk_margin", "entropy", "maskgit_plus"): raise RuntimeError(alg)
    return _GU.sample_tokens(mask_logits, temperature=0.0, top_p=TOP_P, top_k=TOP_K, margin_confidence=(alg == "topk_margin"), neg_entropy=(alg == "entropy"))
TOP_K, TOP_P = 50, 1.0


@torch.inference_mode()
def generate(m, id_lists, fns=None, steps=STEPS, alg="topk_margin", canvas=CANVAS, eps=1e-3, mode="orig", tau=0.5, cmax=2, defer_k=2, return_canvas=False):
    m.fns = fns or {}; m.step = -1
    inp = torch.tensor(id_lists, device="cuda"); B, L = inp.shape; mid_ = m.mask_id
    x = F.pad(inp, (0, canvas), value=mid_)
    ts = torch.linspace(1, eps, steps + 1, device=x.device)
    SPt = torch.tensor(sorted(m.SP), device=x.device)
    rcount = torch.zeros(B, canvas, dtype=torch.long, device=x.device); flips = torch.zeros(B, canvas, dtype=torch.long, device=x.device)
    prev_arg = None; prev_mask = None; n_remask = torch.zeros(B, dtype=torch.long, device=x.device); n_defer = torch.zeros(B, dtype=torch.long, device=x.device)
    rows_ = torch.arange(B, device=x.device)
    for i in range(steps):
        mask_index = (x == mid_)
        logits = m.model(x, "full", None).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        mask_logits = logits[mask_index]
        t = ts[i]; s = ts[i + 1]
        confidence, x0 = _scores(mask_logits, alg)
        full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
        full_confidence[mask_index] = confidence
        if mode == "defer":
            arg = torch.full_like(x, -1); arg[mask_index] = x0; ac = arg[:, L:]; mc = mask_index[:, L:]
            if prev_arg is not None: flips += ((ac != prev_arg) & mc & prev_mask).long()
            prev_arg, prev_mask = ac.clone(), mc.clone()
            dmask = mc & (flips >= defer_k) & ~torch.isin(ac, SPt)
            if i < steps - 1 and dmask.any():
                fc = full_confidence[:, L:]; fc[dmask] = fc[dmask] - 10.0; full_confidence[:, L:] = fc; n_defer += dmask.any(1).long()
        if mode in ("orig", "defer"):
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            ntt = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            if ntt > 0:
                _, transfer_index = torch.topk(full_confidence, ntt)
                x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mid_; x_[mask_index] = x0.clone()
                ri = rows_.unsqueeze(1).expand_as(transfer_index); x[ri, transfer_index] = x_[ri, transfer_index]
        else:
            nm = mask_index.sum(1)
            kr = [int(nm[r] * (1 - s / t)) if i < steps - 1 else int(nm[r]) for r in range(B)]; kmax = max(kr)
            if kmax > 0:
                _, transfer_index = torch.topk(full_confidence, kmax)
                x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mid_; x_[mask_index] = x0.clone()
                valid = torch.arange(kmax, device=x.device)[None, :] < torch.tensor(kr, device=x.device)[:, None]
                ri = rows_.unsqueeze(1).expand_as(transfer_index)
                x[ri[valid], transfer_index[valid]] = x_[ri[valid], transfer_index[valid]]
            if i < steps - 1:
                # 이번 forward 에서 보이던 (= 이전 step 까지 확정된) 답 내용 칸: 캔버스 안, 특수 토큰 아님, 첫 특수 토큰보다 앞
                vis = ~mask_index[:, L:]; tok = x[:, L:]; isp = torch.isin(tok, SPt) & vis
                first_sp = torch.where(isp.any(1), isp.float().argmax(1), torch.full((B,), canvas, device=x.device))
                content = vis & ~isp & (torch.arange(canvas, device=x.device)[None, :] < first_sp[:, None]) & (rcount < cmax)
                if content.any():
                    lp = torch.softmax(logits[:, L:][content].float(), -1).gather(-1, tok[content].unsqueeze(-1)).squeeze(-1)
                    flag = torch.zeros_like(content); flag[content] = lp < tau
                    if flag.any():
                        xc = x[:, L:]; xc[flag] = mid_; x[:, L:] = xc; rcount += flag.long(); n_remask += flag.sum(1)
    m.fns = {}
    out = []
    for j in range(B):
        gen = x[j, L:].tolist(); gen = gen[: next((q for q, t_ in enumerate(gen) if t_ in m.SP), len(gen))]; out.append(extract_answer(m.tok.decode(gen, skip_special_tokens=True)))
    stats = {"n_remask": n_remask.tolist(), "n_defer_steps": n_defer.tolist()}
    return (out, x[:, L:].clone().cpu(), stats) if return_canvas else (out, stats)


@torch.inference_mode()
def resume(m, id_lists, canvases, reset, steps=8, fns=None, alg="topk_margin", eps=1e-3):
    """E21: 완성된 첫 생성 캔버스 canvases [B, canvas] (확정 토큰) 에서 reset [B, canvas] (bool) 칸만 다시 가리고 steps 동안 다시 채운다.
    옮기는 개수는 행별 마스크 수 기준 (원본 규칙 int(n*(1-s/t)), 마지막 step 에 남은 전부).  나머지 칸은 확정 상태 유지.
    fns 는 common.Dream 의 층 훅 (모든 forward 에 적용).  반환: 답 문자열 목록, 최종 캔버스 (cpu)."""
    m.fns = fns or {}; m.step = -1
    inp = torch.tensor(id_lists, device="cuda"); B, L = inp.shape; mid_ = m.mask_id; canvas = canvases.shape[1]
    x = torch.cat([inp, canvases.to("cuda").long()], 1); xc = x[:, L:]; xc[reset.to("cuda")] = mid_; x[:, L:] = xc
    ts = torch.linspace(1, eps, steps + 1, device=x.device); rows_ = torch.arange(B, device=x.device)
    for i in range(steps):
        mask_index = (x == mid_)
        if not mask_index.any(): break
        logits = m.model(x, "full", None).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        mask_logits = logits[mask_index]
        t = ts[i]; s = ts[i + 1]
        confidence, x0 = _scores(mask_logits, alg)
        full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype); full_confidence[mask_index] = confidence
        nm = mask_index.sum(1)
        kr = [int(nm[r] * (1 - s / t)) if i < steps - 1 else int(nm[r]) for r in range(B)]; kmax = max(kr)
        if kmax > 0:
            _, transfer_index = torch.topk(full_confidence, kmax)
            x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mid_; x_[mask_index] = x0.clone()
            valid = torch.arange(kmax, device=x.device)[None, :] < torch.tensor(kr, device=x.device)[:, None]
            ri = rows_.unsqueeze(1).expand_as(transfer_index)
            x[ri[valid], transfer_index[valid]] = x_[ri[valid], transfer_index[valid]]
    m.fns = {}
    out = []
    for j in range(B):
        gen = x[j, L:].tolist(); gen = gen[: next((q for q, t_ in enumerate(gen) if t_ in m.SP), len(gen))]; out.append(extract_answer(m.tok.decode(gen, skip_special_tokens=True)))
    return out, x[:, L:].clone().cpu()
