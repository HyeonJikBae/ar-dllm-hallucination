"""E27 (PREREG.md 'E27 먼저 확정된 칸의 영향').  a: 순전파 분석 (생성 없음), b: 칸 0 을 step 0 argmax 로 먼저 확정하고 생성.
usage: python followup_mk/e27.py --part a|b|ab [--limit N]"""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import dloop_p2 as dl
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
PART = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "ab"
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); w7 = {int(r): j for j, r in enumerate(R7)}; meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r"))
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
tok0 = {int(r): int(committed[j, 0]) for j, r in enumerate(R7)}
m = Dream(); SP = m.SP; MID = m.mask_id; relx = {i: rows[i]["relation"] for i in range(len(rows))}
S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}; ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}
sfx = "_pilot" if LIM else ""
isc = lambda r: meta[r]["new_type"] == "correct"
@torch.inference_mode()
def slot_logits(xs):
    """xs: LongTensor [B, L+32] -> 칸 0 로짓 [B, V] (float32).  Dream 은 로짓을 한 칸 밀므로 칸 j 의 로짓은 위치 L+j (dloop 과 같은 처리)."""
    m.fns = {}; lg = m.model(xs, "full", None).logits; lg = torch.cat([lg[:, :1], lg[:, :-1]], dim=1); return lg
# ================= E27a =================
if "a" in PART:
    ok_tok = lambda t: t not in SP and t != MID
    byfact = collections.defaultdict(list)
    for r in R7:
        r = int(r)
        if r in meta and ok_tok(tok0[r]): byfact[fid[r]].append(r)
    def pick(pool, seed): return int(pool[np.random.default_rng(seed).integers(len(pool))]) if pool else None
    items = []   # (kind, set, row, gold_tok)
    for s, i in S:
        if i not in meta: continue
        if s in ("broken", "entity") and not isc(i):
            dj = pick([j for j in byfact[fid[i]] if j != i and isc(j)], 11 * i + 1)   # E20 와 같은 공여 정의
            if dj is not None: items.append(("wrong", s, i, tok0[dj]))
        elif s == "correct" and isc(i) and ok_tok(tok0[i]): items.append(("control", s, i, tok0[i]))
    print(f"E27a 대상: 오답 {sum(k_ == 'wrong' for k_, *_ in items)}, 정답 대조 {sum(k_ == 'control' for k_, *_ in items)}", flush=True)
    res = []; t00 = time.time(); sanity = [0, 0]
    for n_, (kind, s, i, gold) in enumerate(items):
        j = w7[i]; t0 = int(lastm[j, 0]); L = len(ids[i])
        canvas = np.where(MK[j, t0], MID, committed[j]); cand = [c for c in range(1, 32) if not MK[j, t0, c]]
        base = np.array(ids[i] + canvas.tolist()); vs = [base.copy()]
        for c in cand: v = base.copy(); v[L + c] = MID; vs.append(v)
        v = base.copy(); v[L + 1:] = MID; vs.append(v)                    # 확정된 칸 전부 가림 (칸 0 은 원래 가려져 있음) = step 0 상태
        xs = torch.tensor(np.stack(vs), device="cuda"); top1 = []; rank = []; lp = []
        for b0 in range(0, len(xs), 48):
            lg = slot_logits(xs[b0:b0 + 48])[:, L].float(); top1 += lg.argmax(-1).tolist(); rank += (lg > lg[:, gold:gold + 1]).sum(-1).tolist(); lp += torch.log_softmax(lg, -1)[:, gold].tolist()
        sanity[1] += 1; sanity[0] += int(top1[0] == int(committed[j, 0]))
        flips_gold = [c for c, t in zip(cand, top1[1:-1]) if t == gold]; flips_any = [c for c, t in zip(cand, top1[1:-1]) if t != top1[0]]
        res.append(dict(kind=kind, set=s, row=i, t0=t0, gold=gold, n_cand=len(cand), base_top1=top1[0], base_is_gold=bool(top1[0] == gold), base_rank=rank[0], base_logp=lp[0],
                        flips_gold=flips_gold, flips_any=flips_any, all_top1=top1[-1], all_is_gold=bool(top1[-1] == gold), all_rank=rank[-1], all_logp=lp[-1],
                        cand=cand, dlogp=[round(a - lp[0], 3) for a in lp[1:-1]], cand_special=[int(committed[j, c]) in SP for c in cand]))
        if (n_ + 1) % 100 == 0: print(f"  {n_ + 1}/{len(items)} {time.time() - t00:.0f}s", flush=True)
    json.dump(dict(sanity_base_top1_E7일치=sanity, rows=res), open(f"{OUT}/results_e27a{sfx}.json", "w"), ensure_ascii=False)
    print(f"E27a 끝: sanity (t0 입력의 칸 0 argmax 가 E7 확정 토큰과 같음) {sanity[0]}/{sanity[1]}, {time.time() - t00:.0f}s", flush=True)
# ================= E27b =================
if "b" in PART:
    fo = open(f"{OUT}/results_e27b{sfx}.jsonl", "w", encoding="utf-8"); t0_ = time.time(); nsame = ntot = 0
    for L, ix in by_length([i for _, i in S], lambda i: ids[i]):
        b = [ids[i] for i in ix]; xs = torch.tensor([x + [MID] * CANVAS for x in b], device="cuda")
        pf = slot_logits(xs)[:, L].argmax(-1).tolist()
        base_a, _ = dl.generate(m, b); pf_a, _ = dl.generate(m, b, prefill=pf)
        for i, a0, a1, p in zip(ix, base_a, pf_a, pf):
            same = (p == tok0.get(i, -1)); nsame += same; ntot += 1
            fo.write(json.dumps({"cond": "base", "set": st[i], "row": i, "k": int(k[i]), "answer": a0, "new_type": m.grade(i, a0)}, ensure_ascii=False) + "\n")
            fo.write(json.dumps({"cond": "칸0먼저", "set": st[i], "row": i, "k": int(k[i]), "answer": a1, "new_type": m.grade(i, a1), "step0_tok": int(p), "step0_tok_is_E7_tok": bool(same), "step0_tok_special": bool(p in SP)}, ensure_ascii=False) + "\n")
        fo.flush()
    print(f"E27b 끝: step 0 칸 0 argmax 가 E7 첫 생성의 칸 0 토큰과 같은 비율 {nsame}/{ntot}, {time.time() - t0_:.0f}s", flush=True)
print("DONE", flush=True)
