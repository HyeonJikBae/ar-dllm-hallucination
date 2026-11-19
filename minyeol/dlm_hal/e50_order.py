"""E50 (PREREG.md 'E50 디코딩 순서 x 경로, 전체 규모').  repo 루트에서 실행.
행: e7_rows() 9,950 (1~4/5 5,950 + 0/5 2,000 + 5/5 2,000).  묶음은 E7 과 같은 by_length (R7 순서, 16 행).
조건 (같은 묶음, 따로 생성):
  base    : topk_margin, 원래 순서 (dloop.py mode "orig" 와 같은 코드)
  칸0먼저 : E27b 와 같은 구현.  step 0 forward (전부 마스크) 의 칸 0 argmax 를 답 칸 0 에 넣고 (prefill) 32 step 을 새로 돈다.
            칸 0 은 step 0 입력부터 보이고 끝까지 유지.  argmax 는 base 의 step 0 forward 로짓에서 가져온다 (E27b 의 따로 한 forward 와 같은 입력).
  L2R     : base 와 같은 step 별 확정 개수 (배치 평균 마스크 수 x (1 - s/t), 마지막 step 은 전부) 를, 가려진 칸 중 가장 왼쪽부터 확정 (각 칸의 argmax).
            (dloop_p4.py mode "ar" 는 매 step 한 칸씩 (step 0 포함) 확정해 base 일정과 다르므로 쓰지 않음.)
  entropy : alg="entropy", 원래 순서.
저장 (행 x 조건): answer, grade, tok0 (최종 칸 0 토큰), slot0_step (칸 0 이 확정된 step; 칸0먼저는 0), canvas (최종 32 칸), csteps (칸별 확정 step),
  base 에만 gold_tok (tok(" " + gold_value)[0]), gold_rank0 (step 0 칸 0 실제 로짓 기준 금 순위, 0 = 1 위), arg0 (step 0 칸 0 argmax).
usage: python followup_mk/e50_order.py --pilot            (E7 묶음 2 개 = 32 행, dloop orig 와 E7 대조)
       python followup_mk/e50_order.py --shard s --nshard n (묶음 번호 % n == s)
"""
import collections, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "followup_mk")
from common import *
import dloop as dl

PILOT = "--pilot" in sys.argv
SH = int(sys.argv[sys.argv.index("--shard") + 1]) if "--shard" in sys.argv else 0
NSH = int(sys.argv[sys.argv.index("--nshard") + 1]) if "--nshard" in sys.argv else 1
CONDS = ["base", "칸0먼저", "L2R", "entropy"]


@torch.inference_mode()
def gen(m, id_lists, alg="topk_margin", order="conf", prefill=None, gold=None, steps=STEPS, canvas=CANVAS, eps=1e-3):
    """dloop.generate (mode orig) 복사 + order="l2r", prefill, 칸별 확정 step, step 0 칸 0 argmax / 금 순위."""
    m.fns = {}; m.step = -1
    inp = torch.tensor(id_lists, device="cuda"); B, L = inp.shape; mid_ = m.mask_id
    x = F.pad(inp, (0, canvas), value=mid_)
    cst = torch.full((B, canvas), -1, dtype=torch.long, device=x.device)
    if prefill is not None:
        pf = torch.tensor(prefill, device=x.device); sel = pf >= 0; x[sel, L] = pf[sel]; cst[sel, 0] = 0
    ts = torch.linspace(1, eps, steps + 1, device=x.device)
    rows_ = torch.arange(B, device=x.device); negpos = -torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
    arg0 = rank0 = None; sched = []
    for i in range(steps):
        mask_index = (x == mid_)
        logits = m.model(x, "full", None).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        if i == 0:
            lf = logits[:, L].float(); arg0 = lf.argmax(-1).tolist()
            if gold is not None:
                gt = torch.tensor(gold, device=x.device); rank0 = (lf > lf.gather(1, gt[:, None])).sum(1).tolist()
        mask_logits = logits[mask_index]
        t = ts[i]; s = ts[i + 1]
        confidence, x0 = dl._scores(mask_logits, alg)
        if order == "l2r":   # 가려진 칸만, 왼쪽일수록 높은 점수 (float32, 동점 없음)
            full_confidence = torch.where(mask_index, negpos[None, :].expand(B, -1), torch.full((B, x.shape[1]), -torch.inf, device=x.device))
        else:
            full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
        num_mask_token = mask_index.sum() / mask_index.shape[0]
        ntt = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
        sched.append(ntt)
        if ntt > 0:
            _, transfer_index = torch.topk(full_confidence, ntt)
            x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mid_; x_[mask_index] = x0.clone()
            ri = rows_.unsqueeze(1).expand_as(transfer_index); x[ri, transfer_index] = x_[ri, transfer_index]
            ok = transfer_index >= L; cst[ri[ok], transfer_index[ok] - L] = i
    m.fns = {}
    out = []
    for j in range(B):
        g_ = x[j, L:].tolist(); g_ = g_[: next((q for q, t_ in enumerate(g_) if t_ in m.SP), len(g_))]; out.append(extract_answer(m.tok.decode(g_, skip_special_tokens=True)))
    return dict(answers=out, canvas=x[:, L:].tolist(), csteps=cst.tolist(), arg0=arg0, rank0=rank0, sched=sched)


def run_batch(m, ix, ids):
    b = [ids[i] for i in ix]
    gold = [m.tok(" " + rows[i]["gold_value"])["input_ids"][0] for i in ix]
    R = {}
    R["base"] = gen(m, b, gold=gold)
    R["칸0먼저"] = gen(m, b, prefill=R["base"]["arg0"])
    R["L2R"] = gen(m, b, order="l2r")
    R["entropy"] = gen(m, b, alg="entropy")
    return R, gold


def records(m, ix, R, gold, setname):
    recs = []
    for c in CONDS:
        r_ = R[c]
        for j, i in enumerate(ix):
            a = r_["answers"][j]
            d = {"cond": c, "row": int(i), "k": int(k[i]), "set": setname[i], "answer": a, "grade": m.grade(i, a), "tok0": int(r_["canvas"][j][0]),
                 "slot0_step": int(r_["csteps"][j][0]), "canvas": r_["canvas"][j], "csteps": r_["csteps"][j]}
            if c == "base": d.update(gold_tok=int(gold[j]), gold_rank0=int(r_["rank0"][j]), arg0=int(r_["arg0"][j]))
            if c == "칸0먼저": d.update(prefill_tok=int(R["base"]["arg0"][j]), prefill_special=bool(R["base"]["arg0"][j] in m.SP))
            recs.append(d)
    return recs


def main():
    R7, Z7, F7 = e7_rows(); zs, fs = set(Z7.tolist()), set(F7.tolist())
    setname = {int(i): ("zero" if int(i) in zs else "five" if int(i) in fs else "mid") for i in R7}
    m = Dream(); ids = {int(i): m.ids(build_prompt(rows[int(i)]["question"])) for i in R7}
    batches = list(by_length([int(i) for i in R7], lambda i: ids[i]))
    print(f"묶음 {len(batches)} 개, 행 {sum(len(x) for _, x in batches)}", flush=True)
    if PILOT:
        full = [b for b in range(len(batches)) if len(batches[b][1]) == BS]
        pick = [full[len(full) // 3], full[2 * len(full) // 3]]
        meta = {}
        for l in open(f"{OUT}/e7/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
        IDS = np.asarray(np.load(f"{OUT}/e7/ids.npy", mmap_mode="r")); MK = np.asarray(np.load(f"{OUT}/e7/mask.npy", mmap_mode="r"))
        w7 = {int(r): j for j, r in enumerate(R7)}
        fo = open(f"{OUT}/results_e50_pilot.jsonl", "w", encoding="utf-8"); chk = collections.Counter()
        for b in pick:
            L, ix = batches[b]; t0 = time.time()
            R, gold = run_batch(m, ix, ids)
            ref, ref_canvas, _ = dl.generate(m, [ids[i] for i in ix], return_canvas=True)
            print(f"  묶음 {b} (L={L}, n={len(ix)}): {time.time() - t0:.1f}s; 일정 base {R['base']['sched']}\n    칸0먼저 {R['칸0먼저']['sched']}\n    L2R {R['L2R']['sched']}", flush=True)
            for j, i in enumerate(ix):
                chk["n"] += 1
                chk["base==dloop 답"] += R["base"]["answers"][j] == ref[j]; chk["base==dloop 캔버스"] += R["base"]["canvas"][j] == ref_canvas[j].tolist()
                chk["base==E7 답"] += R["base"]["answers"][j] == meta[i]["answer"]
                jj = w7[i]; lastm = int(np.where(MK[jj, :, 0])[0].max()); chk["base 칸0 토큰==E7"] += R["base"]["canvas"][j][0] == int(IDS[jj, lastm, 0])
                chk["base 칸0 확정 step==E7 (마지막 마스크 step+?)"] += R["base"]["csteps"][j][0] == lastm
                cs = R["L2R"]["csteps"][j]; chk["L2R 칸 순서 단조"] += all(cs[q] <= cs[q + 1] for q in range(31))
                chk["칸0먼저 칸0==step0 argmax"] += R["칸0먼저"]["canvas"][j][0] == R["base"]["arg0"][j]
                chk["base grade==E7 grade"] += m.grade(i, R["base"]["answers"][j]) == meta[i]["new_type"]
                for c in CONDS: chk[f"정답 {c}"] += m.grade(i, R[c]["answers"][j]) == "correct"
            # L2R 확정 개수가 base 일정과 같은지
            chk["L2R 일정==base 일정 (묶음)"] += R["L2R"]["sched"] == R["base"]["sched"]
            for d in records(m, ix, R, gold, setname): fo.write(json.dumps(d, ensure_ascii=False) + "\n")
        fo.close()
        print(json.dumps(chk, ensure_ascii=False, indent=1), flush=True)
        for b in pick:
            L, ix = batches[b]
            for i in ix[:4]: print(f"   row {i} k={k[i]} gold={rows[i]['gold_value']!r} E7={meta[i]['answer']!r}", flush=True)
        print("DONE", flush=True); return
    fout = f"{OUT}/results_e50_s{SH}.jsonl"; have = collections.defaultdict(set)
    if os.path.exists(fout):
        for l in open(fout, encoding="utf-8"):
            try: d = json.loads(l); have[d["cond"]].add(d["row"])
            except Exception: pass
    mine = [b for b in range(len(batches)) if b % NSH == SH]
    todo = [b for b in mine if not all(i in have[c] for c in CONDS for i in batches[b][1])]
    print(f"shard {SH}/{NSH}: 묶음 {len(mine)}, 남은 묶음 {len(todo)}", flush=True)
    fo = open(fout, "a", encoding="utf-8"); t0 = time.time(); cnt = 0
    for n_, b in enumerate(todo):
        L, ix = batches[b]
        R, gold = run_batch(m, ix, ids)
        recs = [d for d in records(m, ix, R, gold, setname) if d["row"] not in have[d["cond"]]]
        fo.write("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in recs)); fo.flush(); cnt += len(ix)
        if (n_ + 1) % 20 == 0: print(f"  묶음 {n_ + 1}/{len(todo)}, 행 {cnt}, {time.time() - t0:.0f}s", flush=True)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
