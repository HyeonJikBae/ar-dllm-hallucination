"""E55 DiffuLLaMA-7B (diffusionfamily/diffullama, LLaMA-2-7B 에서 적응) 확산 샘플러와 전체 실행.
샘플러 = followup_mk/dloop.py 의 "orig" 고리 (Dream _sample, temperature 0) 를 DiffuLLaMA 에 옮긴 것:
  - 양방향 주의: HF LlamaForCausalLM 에 4D 전부-참 마스크를 넘김 (HKUNLP/DiffuLLaMA 의 attn_mask_ratio=1.0 annealed mask 와 같음).
  - 로짓 정렬: DiffuLLaMA 도 AR 처럼 '위치 i 출력 = i+1 토큰 예측' (shift) -> Dream 과 같게 logits = cat(lg[:, :1], lg[:, :-1]).  (--test 로 확인)
  - 확신 = Dream sample_tokens 와 같은 top_k=50 거른 뒤 softmax 의 top1 - top2 (topk_margin), bf16 그대로.
  - 옮기는 개수 = Dream 과 같은 linspace(1, 1e-3, 33) 일정 (step 0 은 0 개, 마지막 step 에 나머지 전부).
  - 마스크 토큰 = tokenizer.mask_token ('ти', id 811), 답 끝 = 첫 특수 토큰 (all_special_ids) 앞까지, extract_answer.
조건 (같은 묶음): base, 칸0먼저 (E27b 와 같음: base step 0 forward 의 칸 0 로짓 argmax 를 step 0 입력부터 칸 0 에 넣고 나머지는 원래대로).
usage: python followup_mk/e55_dllm.py --test            (정렬·양방향·예시 확인)
       python followup_mk/e55_dllm.py --run [--shard i --nshard n] [--limit N] [--bs 64] [--cond base|both] [--rows rows.json --tag _B]
  A 단계: --cond base, 55,225 행 전부 (k/5 를 정함).  B 단계: --cond both --rows (A 의 1~4/5 사실 행) --tag _B, base 와 칸0먼저를 같은 묶음에서.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "followup_mk")
from e55_common import rows, build_prompt, grade, gold_first, groups, decode_answer, oom_retry

MODEL = "diffusionfamily/diffullama"; CANVAS, STEPS, TOP_K, TOP_P = 32, 32, 50, 1.0


def top_k_logits(logits, top_k):   # Dream generation_utils 와 같음
    top_k = min(top_k, logits.size(-1)); rm = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(rm, torch.finfo(logits.dtype).min)

def sample_tokens(logits, margin=True):   # Dream sample_tokens(temperature=0, top_p=1.0, top_k=50, margin_confidence=True)
    probs = torch.softmax(top_k_logits(logits, TOP_K), dim=-1); confidence, x0 = probs.max(dim=-1)
    if margin:   # Dream 은 전체 정렬 후 [0]-[1]; topk(2) 는 같은 두 값 (메모리 절약)
        sp = probs.topk(2, dim=-1).values; confidence = sp[:, 0] - sp[:, 1]
    return confidence, x0


class DLLM:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        self.SP = set(self.tok.all_special_ids); self.mask_id = self.tok.mask_token_id
        assert self.mask_id == 811, self.mask_id
    @torch.inference_mode()
    def raw_logits(self, x, causal=False):
        B, T = x.shape
        am = torch.ones(B, 1, T, T, dtype=torch.bool, device=x.device)
        if causal: am = torch.tril(am)
        return self.model(input_ids=x, attention_mask=am, use_cache=False).logits
    def logits(self, x): lg = self.raw_logits(x); return torch.cat([lg[:, :1], lg[:, :-1]], dim=1)
    @torch.inference_mode()
    def canvas_logits(self, x, canvas):
        """캔버스 칸 j 의 (shift 된) 로짓 = 위치 L-1+j 출력.  lm_head 는 끝 canvas+1 위치에만 (logits_to_keep) -> [B, canvas, V]."""
        B, T = x.shape; am = torch.ones(B, 1, T, T, dtype=torch.bool, device=x.device)
        return self.model(input_ids=x, attention_mask=am, use_cache=False, logits_to_keep=canvas + 1).logits[:, :canvas]


@torch.inference_mode()
def generate(m, id_lists, prefill=None, gold=None, steps=STEPS, canvas=CANVAS, eps=1e-3):
    """dloop.generate (mode orig) 와 같은 고리.  gold: 행별 금 첫 토큰 (칸 0 순위 기록용).  반환 (답, 기록 dict)."""
    inp = torch.tensor(id_lists, device="cuda"); B, L = inp.shape; mid_ = m.mask_id
    x = F.pad(inp, (0, canvas), value=mid_)
    if prefill is not None:
        pf = torch.tensor(prefill, device=x.device); sel = pf >= 0; x[sel, L] = pf[sel]
    ts = torch.linspace(1, eps, steps + 1, device=x.device); rows_ = torch.arange(B, device=x.device)
    g = torch.tensor(gold if gold is not None else [0] * B, device=x.device)
    cstep = torch.full((B, canvas), -1, dtype=torch.long, device=x.device); cstep[x[:, L:] != mid_] = -2   # -2 = 처음부터 채움 (prefill)
    s0_rank = torch.full((B, steps), -1, dtype=torch.long, device=x.device); s0_arg = torch.full((B, steps), -1, dtype=torch.long, device=x.device)
    rec = {}; masked = (x[:, L:] == mid_)   # 마스크 여부는 캔버스 안에서 따로 추적 ('ти' 가 실제 어휘라 토큰 비교 대신)
    for i in range(steps):
        logits = m.canvas_logits(x, canvas)   # [B, canvas, V], 칸 j = 위치 L-1+j (Dream 과 같은 shift)
        z0 = logits[:, 0].float(); gv = z0[rows_, g]
        s0_rank[:, i] = (z0 > gv[:, None]).sum(-1); s0_arg[:, i] = z0.argmax(-1)
        if i == 0:
            rec["s0_gold_logit0"] = gv.tolist(); rec["s0_top1_logit0"] = z0.max(-1).values.tolist()
            rec["s0_argmax0"] = z0.argmax(-1).tolist()   # 칸0먼저에 넣을 토큰 (E27b: 로짓 argmax)
        mask_logits = logits[masked]
        t = ts[i]; s = ts[i + 1]
        confidence, x0 = sample_tokens(mask_logits); del logits, mask_logits
        full_confidence = torch.full(masked.shape, -torch.inf, device=x.device, dtype=confidence.dtype); full_confidence[masked] = confidence
        num_mask_token = masked.sum() / masked.shape[0]
        ntt = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
        if ntt > 0:
            _, transfer_index = torch.topk(full_confidence, ntt)
            x_ = torch.full_like(masked, mid_, dtype=torch.long); x_[masked] = x0.clone()
            ri = rows_.unsqueeze(1).expand_as(transfer_index); xc = x[:, L:].clone(); xc[ri, transfer_index] = x_[ri, transfer_index]; x[:, L:] = xc
            masked[ri, transfer_index] = False
        newly = ~masked & (cstep == -1); cstep[newly] = i
    out = [decode_answer(m.tok, x[j, L:].tolist(), m.SP) for j in range(B)]
    rec.update(canvas=x[:, L:].tolist(), commit_step=cstep.tolist(), s0_rank=s0_rank.tolist(), s0_arg=s0_arg.tolist())
    return out, rec


def run_test(m):
    tok = m.tok; res = {}
    # (1) 로짓 정렬: 문장 안 한 토큰을 가리고, 그 토큰을 '앞 위치 출력' (shift) 과 '자기 위치 출력' (no shift) 중 어느 쪽이 맞히는지
    sents = ["The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome.",
             "Barack Obama was the 44th president of the United States, and he was born in Hawaii.",
             "Water boils at 100 degrees Celsius at sea level, and it freezes at 0 degrees Celsius.",
             "The Eiffel Tower is located in Paris, the capital city of France, and was built in 1889.",
             "William Shakespeare wrote Hamlet, Macbeth, Othello and King Lear among many other plays."]
    hit = {"shift": 0, "noshift": 0}; n = 0
    for s_ in sents:
        ids = tok(s_)["input_ids"]
        for j in range(2, len(ids) - 1):
            x = torch.tensor([ids], device="cuda"); x[0, j] = m.mask_id; lg = m.raw_logits(x)[0].float()
            hit["shift"] += int(lg[j - 1].argmax() == ids[j]); hit["noshift"] += int(lg[j].argmax() == ids[j]); n += 1
    res["align"] = dict(n=n, **hit); print("정렬 (가린 토큰 1 개 복원 정확도):", res["align"], flush=True)
    # (2) 양방향: 같은 입력에서 뒤 토큰만 바꾸면 앞 위치 로짓이 바뀌는지 (양방향 마스크 대 인과 마스크), 그리고 오른쪽 문맥으로 가린 토큰 맞히기
    ids = tok("The capital of Germany is Berlin.")["input_ids"]; x = torch.tensor([ids], device="cuda")
    x2 = x.clone(); x2[0, -2] = tok("Madrid", add_special_tokens=False)["input_ids"][-1]
    d_bi = (m.raw_logits(x)[0, 2].float() - m.raw_logits(x2)[0, 2].float()).abs().max().item()
    d_ca = (m.raw_logits(x, causal=True)[0, 2].float() - m.raw_logits(x2, causal=True)[0, 2].float()).abs().max().item()
    res["bidir"] = dict(maxdiff_pos2_bidirectional=d_bi, maxdiff_pos2_causal=d_ca); print("뒤 토큰 바꿈 -> 앞 위치 로짓 변화 (양방향/인과):", d_bi, d_ca, flush=True)
    rc = []
    for a, b in [("The capital of", " is Berlin."), ("The capital of", " is Tokyo."), ("The capital of", " is Rome."), ("The president of the", " is Joe Biden.")]:
        pa = tok(a)["input_ids"]; pb = tok(b, add_special_tokens=False)["input_ids"]; ids = pa + [m.mask_id] + pb
        x = torch.tensor([ids], device="cuda"); lg = m.raw_logits(x)[0].float(); lc = m.raw_logits(x, causal=True)[0].float()
        rc.append(dict(left=a, right=b, bidir=tok.decode([int(lg[len(pa) - 1].argmax())]), causal=tok.decode([int(lc[len(pa) - 1].argmax())])))
    res["right_context"] = rc; print("오른쪽 문맥 이용:", rc, flush=True)
    # (3) 실제 질문 프롬프트 예시
    ex = []; idx = [0, 5, 1000, 5000, 12345, 20000, 33333, 40000, 47000, 55000]
    for i in idx:
        p = m.tok(build_prompt(rows[i]["question"]))["input_ids"]; gd = gold_first(m.tok, i, p)
        a, rec = generate(m, [p], gold=[gd]); a1, _ = generate(m, [p], prefill=[rec["s0_argmax0"][0]], gold=[gd])
        full = m.tok.decode(rec["canvas"][0]); ex.append(dict(row=i, q=rows[i]["question"], gold=rows[i]["gold_value"], answer=a[0], grade=grade(i, a[0]), canvas=full,
                                                              s0_rank0=rec["s0_rank"][0][0], s0_commit=rec["commit_step"][0][0], first_commit=[c for c, s in enumerate(rec["commit_step"][0]) if s == 1],
                                                              slot0first=a1[0], grade_s0f=grade(i, a1[0])))
        print(json.dumps(ex[-1], ensure_ascii=False), flush=True)
    res["examples"] = ex
    # (4) 묶음 일관성: 같은 프롬프트를 묶음 1 과 묶음 안에서 돌린 답이 같은지 (bf16)
    json.dump(res, open("followup_mk/e55_sampler_check.json", "w"), ensure_ascii=False, indent=1)


def run(m, args):
    items = list(range(len(rows)))
    if args.rows: items = sorted(int(x) for x in json.load(open(args.rows)))
    if args.limit: items = items[:: max(1, len(items) // args.limit)][: args.limit]
    pids = {i: m.tok(build_prompt(rows[i]["question"]))["input_ids"] for i in items}
    gold = {i: gold_first(m.tok, i, pids[i]) for i in items}
    batches = list(groups(items, lambda i: pids[i], args.bs)); mine = [(j, L, ix) for j, (L, ix) in enumerate(batches) if j % args.nshard == args.shard]
    fn = f"followup_mk/results_e55_dllm{'_pilot' if args.limit else ''}{args.tag}_s{args.shard}.jsonl"
    done = set()
    if os.path.exists(fn):
        for l in open(fn):
            try: done.add(json.loads(l)["row"])
            except Exception: pass
    fo = open(fn, "a", encoding="utf-8"); t0 = time.time(); nb = 0
    print(f"shard {args.shard}/{args.nshard}: 묶음 {len(mine)}, 이미 {len(done)} 행", flush=True)
    for j, L, ix in mine:
        if all(i in done for i in ix): continue
        b = [pids[i] for i in ix]; gd = [max(gold[i], 0) for i in ix]
        a0, r0 = oom_retry(lambda: generate(m, b, gold=gd)); pf = r0["s0_argmax0"]
        if args.cond == "both": a1, r1 = oom_retry(lambda: generate(m, b, prefill=pf, gold=gd))
        for q, i in enumerate(ix):
            if i in done: continue
            o = dict(row=i, batch=j, L=L, bsz=len(ix), gold_id=gold[i],
                     base=dict(answer=a0[q], type=grade(i, a0[q]), canvas=r0["canvas"][q], commit_step=r0["commit_step"][q], s0_rank=r0["s0_rank"][q], s0_arg=r0["s0_arg"][q],
                               s0_gold_logit0=round(r0["s0_gold_logit0"][q], 3), s0_top1_logit0=round(r0["s0_top1_logit0"][q], 3)))
            if args.cond == "both":
                o["s0f"] = dict(answer=a1[q], type=grade(i, a1[q]), tok=int(pf[q]), tok_special=bool(pf[q] in m.SP), canvas=r1["canvas"][q], s0_rank=r1["s0_rank"][q][:1])
            fo.write(json.dumps(o, ensure_ascii=False) + "\n")
        fo.flush(); nb += 1
        if nb % 20 == 0: print(f"  {nb}/{len(mine)} 묶음, {time.time() - t0:.0f}s", flush=True)
    print(f"끝: {time.time() - t0:.0f}s", flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--cond", default="both", choices=["base", "both"]); ap.add_argument("--rows", default=""); ap.add_argument("--tag", default="")
    args = ap.parse_args(); m = DLLM()
    if args.test: run_test(m)
    if args.run: run(m, args)
