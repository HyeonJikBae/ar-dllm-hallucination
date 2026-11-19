"""E56 3 단계 (Dream-v0-Base-7B): TriviaQA 다섯 문구 전 행, 같은 묶음에서 두 조건.
  base    : topk_margin, 32 step, 캔버스 32, temperature 0 (dloop.py mode orig 와 같은 코드; E50 gen 복사).
            step 0 forward (캔버스 전부 마스크) 의 칸 0 실제 로짓 (한 칸 민 로짓의 위치 L) 에서
            금 첫 토큰 순위 rank0 (tok(" " + value)[0], 0 = 1 위), 어느 별칭이든 순위 rank0_any (별칭 첫 토큰 집합의 최소 순위), argmax arg0.
            칸 0 이 가려져 있는 동안 step 별 금 순위 (rank_traj, 확정 뒤는 -1).
  칸0먼저 : E27b/E50 과 같음.  base 의 step 0 칸 0 argmax 를 답 칸 0 에 넣고 (prefill) 32 step 을 새로 돈다.
묶음: common.by_length (같은 길이 프롬프트, 16 행).  shard = 묶음 번호 % nshard.  이어 돌리기 (이미 쓴 행의 묶음은 건너뜀).
채점은 하지 않고 답만 저장 (e56_eval.py 에서 채점).
usage (repo 루트): python followup_mk/e56_dream.py [--shard s --nshard n] [--pilot]
  --pilot: 묶음 2 개, base 를 dloop.generate 와 대조 (같은 답·캔버스여야 함).
"""
import json, os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, "followup_mk")
from common import Dream, build_prompt, extract_answer, by_length, STEPS, CANVAS
import dloop as dl
from e56_common import load_items, first_tokens, VERS

PILOT = "--pilot" in sys.argv; STEER = "--steer" in sys.argv
import numpy as np
from common import add_end
SH = int(sys.argv[sys.argv.index("--shard") + 1]) if "--shard" in sys.argv else 0
NSH = int(sys.argv[sys.argv.index("--nshard") + 1]) if "--nshard" in sys.argv else 1


@torch.inference_mode()
def gen(m, id_lists, prefill=None, gold=None, gsets=None, steps=STEPS, canvas=CANVAS, eps=1e-3, alg="topk_margin", fns=None, cap=False):
    m.fns = fns or {}; m.step = -1
    feats = None; preds = []
    inp = torch.tensor(id_lists, device="cuda"); B, L = inp.shape; mid_ = m.mask_id
    x = F.pad(inp, (0, canvas), value=mid_)
    cst = torch.full((B, canvas), -1, dtype=torch.long, device=x.device)
    if prefill is not None:
        pf = torch.tensor(prefill, device=x.device); sel = pf >= 0; x[sel, L] = pf[sel]; cst[sel, 0] = 0
    ts = torch.linspace(1, eps, steps + 1, device=x.device)
    rows_ = torch.arange(B, device=x.device)
    arg0 = rank0 = rank0_any = None; traj = torch.full((B, steps), -1, dtype=torch.long, device=x.device)
    if gold is not None: gt = torch.tensor(gold, device=x.device)
    for i in range(steps):
        mask_index = (x == mid_)
        logits = m.model(x, "full", None).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        if gold is not None:
            lf = logits[:, L].float(); rk = (lf > lf.gather(1, gt[:, None])).sum(1)
            traj[:, i] = torch.where(mask_index[:, L], rk, torch.full_like(rk, -1))
            if i == 0:
                arg0 = lf.argmax(-1).tolist(); rank0 = rk.tolist()
                rank0_any = [int(min((lf[j] > lf[j, t]).sum().item() for t in gsets[j])) for j in range(B)]
        elif i == 0:
            arg0 = logits[:, L].float().argmax(-1).tolist()
        if cap and i == 0:   # E58: step 0 첫 4 칸 top-1 logprob, 칸 0 top1-top2 margin
            tv = torch.log_softmax(logits[:, L:L + 4].float(), -1).topk(2, -1).values
            feats = dict(lp4=tv[:, :, 0].tolist(), margin0=(tv[:, 0, 0] - tv[:, 0, 1]).tolist())
        mask_logits = logits[mask_index]
        t = ts[i]; s = ts[i + 1]
        confidence, x0 = dl._scores(mask_logits, alg)
        if cap:   # E58 SC: 이 step 의 중간 캔버스 (가려진 칸 = 이번 예측, 확정 칸 = 확정 토큰)
            xp = x.clone(); xp[mask_index] = x0; preds.append(xp[:, L:].cpu())
        full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
        full_confidence[mask_index] = confidence
        num_mask_token = mask_index.sum() / mask_index.shape[0]
        ntt = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
        if ntt > 0:
            _, transfer_index = torch.topk(full_confidence, ntt)
            x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mid_; x_[mask_index] = x0.clone()
            ri = rows_.unsqueeze(1).expand_as(transfer_index); x[ri, transfer_index] = x_[ri, transfer_index]
            ok = transfer_index >= L; cst[ri[ok], transfer_index[ok] - L] = i
    m.fns = {}
    out = []
    for j in range(B):
        g_ = x[j, L:].tolist(); g_ = g_[: next((q for q, t_ in enumerate(g_) if t_ in m.SP), len(g_))]; out.append(extract_answer(m.tok.decode(g_, skip_special_tokens=True)))
    mid_ans = None
    if cap:
        mid_ans = [[None] * steps for _ in range(B)]
        for i_, P in enumerate(preds):
            for j in range(B):
                g_ = P[j].tolist(); g_ = g_[: next((q for q, t_ in enumerate(g_) if t_ in m.SP), len(g_))]; mid_ans[j][i_] = extract_answer(m.tok.decode(g_, skip_special_tokens=True))
    return dict(answers=out, canvas=x[:, L:].tolist(), csteps=cst.tolist(), arg0=arg0, rank0=rank0, rank0_any=rank0_any, traj=traj.tolist(), feats=feats, mid_ans=mid_ans)


def main():
    items = load_items()
    R = [(it["qid"], v) for it in items for v in VERS]; Q = {it["qid"]: it for it in items}
    m = Dream()
    ids = {r: m.ids(build_prompt(Q[r[0]][r[1]])) for r in R}
    ft = {q: first_tokens(m.tok, it["value"], it["aliases"]) for q, it in Q.items()}
    batches = list(by_length(R, lambda r: ids[r]))
    print(f"행 {len(R)}, 묶음 {len(batches)}", flush=True)
    if PILOT:
        full = [b for b in range(len(batches)) if len(batches[b][1]) == 16]; mine = [(b, *batches[b]) for b in (full[len(full) // 3], full[2 * len(full) // 3])]
        fn = "followup_mk/results_e56_dream_pilot.jsonl"
    else:
        mine = [(b, L, ix) for b, (L, ix) in enumerate(batches) if b % NSH == SH]
        fn = f"followup_mk/results_e56_{'steer' if STEER else 'dream'}_s{SH}.jsonl"
    HD = "followup_mk/e56_hid"; os.makedirs(HD, exist_ok=True)
    if STEER:   # E58: 층 24 프롬프트 끝, 겹밖 B (질문 id 5 겹), 0.5 x 평균 ||h||, 모든 step
        Z = np.load("followup_mk/e56_data/e58_dirs.npz"); fold = json.load(open("followup_mk/e56_data/folds.json")); DIR = torch.tensor(Z["B"], device="cuda", dtype=torch.float32); SCALE = float(Z["scale"])
    done = set()
    if os.path.exists(fn):
        for l in open(fn, encoding="utf-8"):
            try: d = json.loads(l); done.add((d["qid"], d["ver"], d["cond"]))
            except Exception: pass
    fo = open(fn, "a", encoding="utf-8"); t0 = time.time(); nb = 0; chk = [0, 0, 0]
    print(f"shard {SH}/{NSH}: 묶음 {len(mine)}, 이미 {len(done)} 기록", flush=True)
    for b, L, ix in mine:
        bl = [ids[r] for r in ix]
        if STEER:
            if all((q, v, "B24") in done for q, v in ix): continue
            vecs = SCALE * DIR[[fold[q] for q, _ in ix]]; rs = gen(m, bl, fns={24: add_end(vecs, L)})
            for j, (q, v) in enumerate(ix): fo.write(json.dumps(dict(qid=q, ver=v, cond="B24", batch=b, answer=rs["answers"][j]), ensure_ascii=False) + "\n")
            fo.flush(); nb += 1
            if nb % 10 == 0: print(f"  {nb}/{len(mine)} 묶음, {time.time() - t0:.0f}s", flush=True)
            continue
        if all((q, v, c) in done for q, v in ix for c in ("base", "칸0먼저")): continue
        gold = [ft[q][0] for q, _ in ix]; gsets = [ft[q][1] for q, _ in ix]; H = {}
        def capf(l_):
            def f(h, step):
                if step == 0: H[l_] = h[:, L - 1].float().cpu().numpy().astype(np.float16)
            return f
        rb = gen(m, bl, gold=gold, gsets=gsets, fns={24: capf(24), 27: capf(27)}, cap=True); rp = gen(m, bl, prefill=rb["arg0"])
        if not PILOT: np.savez(f"{HD}/b{b}.npz", h24=H[24], h27=H[27], qid=np.array([q for q, _ in ix]), ver=np.array([v for _, v in ix]))
        if PILOT:
            ref, refc, _ = dl.generate(m, bl, return_canvas=True)
            for j in range(len(ix)): chk[0] += rb["answers"][j] == ref[j]; chk[1] += rb["canvas"][j] == refc[j].tolist(); chk[2] += 1
        for j, (q, v) in enumerate(ix):
            fo.write(json.dumps(dict(qid=q, ver=v, cond="base", batch=b, L=L, answer=rb["answers"][j], tok0=rb["canvas"][j][0], slot0_step=rb["csteps"][j][0],
                                     gold_tok=gold[j], gold_set=gsets[j], rank0=rb["rank0"][j], rank0_any=rb["rank0_any"][j], arg0=rb["arg0"][j],
                                     rank_traj=rb["traj"][j], canvas=rb["canvas"][j], csteps=rb["csteps"][j],
                                     lp4=rb["feats"]["lp4"][j], margin0=rb["feats"]["margin0"][j], mid_ans=rb["mid_ans"][j]), ensure_ascii=False) + "\n")
            fo.write(json.dumps(dict(qid=q, ver=v, cond="칸0먼저", batch=b, L=L, answer=rp["answers"][j], tok0=rp["canvas"][j][0], slot0_step=rp["csteps"][j][0],
                                     prefill_tok=rb["arg0"][j], prefill_special=bool(rb["arg0"][j] in m.SP), canvas=rp["canvas"][j], csteps=rp["csteps"][j]), ensure_ascii=False) + "\n")
        fo.flush(); nb += 1
        if nb % 10 == 0 or PILOT: print(f"  {nb}/{len(mine)} 묶음, {time.time() - t0:.0f}s", flush=True)
    if PILOT: print(f"pilot 대조: base 답 == dloop {chk[0]}/{chk[2]}, 캔버스 == dloop {chk[1]}/{chk[2]}", flush=True)
    print(f"끝 {time.time() - t0:.0f}s", flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
