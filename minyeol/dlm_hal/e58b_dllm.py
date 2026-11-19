"""E58b (PREREG 'E58 주 방법 확인' (b)): PopQA 위 DiffuLLaMA.  E55 샘플러 (e55_dllm) 와 같은 고리에 층 훅만 더함.
--pass base : 첫 생성.  step 0 칸 0~3 top-1 logprob, 칸 0 top1-top2 로짓 차, 32 step 중간 답 (가려진 칸은 그 step argmax, 확정 칸은 확정 토큰; E30 과 같음) -> SC,
              step 0 프롬프트 끝 상태 (층 27 출력 = B 용, 층 31 출력 = 마지막 층, K/S1 용) 를 e58b/hid_s{shard}.npy 에.
--pass steer: e58b/dirs.npz 의 겹밖 방향 (행의 겹 grp 에 맞는 것) x scale 을 층 27 출력의 프롬프트 끝에 모든 step 더해 생성.  base 와 같은 묶음 (같은 행 목록, 같은 묶기).
              재현 확인: 각 shard 첫 2 묶음은 base 도 다시 돌려 base 결과와 비교.
usage: python followup_mk/e58b_dllm.py --pass base|steer --shard i --nshard n
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "followup_mk")
from e55_common import rows, build_prompt, grade, groups, decode_answer, oom_retry
from e55_dllm import DLLM, sample_tokens, CANVAS, STEPS
from common import grp
LB, LK = 27, 31; OUTD = "followup_mk/e58b"


@torch.inference_mode()
def generate(m, id_lists, add=None, capture=False, steps=STEPS, canvas=CANVAS, eps=1e-3):
    inp = torch.tensor(id_lists, device="cuda"); B, L = inp.shape; mid_ = m.mask_id
    x = F.pad(inp, (0, canvas), value=mid_); ts = torch.linspace(1, eps, steps + 1, device=x.device); rows_ = torch.arange(B, device=x.device)
    st = {"i": 0}; cap = {}
    def hk(li):
        def f(mod, a, out):
            h = out[0] if isinstance(out, tuple) else out
            if capture and st["i"] == 0: cap[li] = h[:, L - 1].float().cpu().numpy()
            if add is not None and li == LB: h[:, L - 1] = (h[:, L - 1].float() + add).to(h.dtype)
        return f
    hs = [m.model.model.layers[li].register_forward_hook(hk(li)) for li in (LB, LK)]
    rec = {}; masked = (x[:, L:] == mid_); inter = [[] for _ in range(B)]
    try:
        for i in range(steps):
            st["i"] = i
            logits = m.canvas_logits(x, canvas)
            if i == 0 and capture:
                lp = torch.log_softmax(logits[:, :4].float(), -1); t2 = logits[:, 0].float().topk(2, -1).values
                rec["lp0"] = lp[:, 0].max(-1).values.tolist(); rec["lp4"] = lp.max(-1).values.mean(-1).tolist(); rec["margin0"] = (t2[:, 0] - t2[:, 1]).tolist()
            mask_logits = logits[masked]; t = ts[i]; s = ts[i + 1]
            confidence, x0 = sample_tokens(mask_logits); del logits, mask_logits
            if capture:
                fill = x[:, L:].clone(); fill[masked] = x0; fl = fill.tolist()
                for j in range(B): inter[j].append(decode_answer(m.tok, fl[j], m.SP))
            full_confidence = torch.full(masked.shape, -torch.inf, device=x.device, dtype=confidence.dtype); full_confidence[masked] = confidence
            num_mask_token = masked.sum() / masked.shape[0]
            ntt = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            if ntt > 0:
                _, ti = torch.topk(full_confidence, ntt)
                x_ = torch.full_like(masked, mid_, dtype=torch.long); x_[masked] = x0.clone()
                ri = rows_.unsqueeze(1).expand_as(ti); xc = x[:, L:].clone(); xc[ri, ti] = x_[ri, ti]; x[:, L:] = xc; masked[ri, ti] = False
    finally:
        for h_ in hs: h_.remove()
    out = [decode_answer(m.tok, x[j, L:].tolist(), m.SP) for j in range(B)]
    if capture: rec.update(inter=inter, hB=cap[LB], hK=cap[LK])
    return out, rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pass", dest="ps", required=True, choices=["base", "steer"]); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()
    m = DLLM(); R = json.load(open(f"{OUTD}/rows.json"))["rows"]
    if args.limit: R = R[:: len(R) // args.limit][: args.limit]
    pids = {i: m.tok(build_prompt(rows[i]["question"]))["input_ids"] for i in R}
    batches = list(groups(sorted(R), lambda i: pids[i], args.bs)); mine = [(j, L, ix) for j, (L, ix) in enumerate(batches) if j % args.nshard == args.shard]
    sfx = f"{'_pilot' if args.limit else ''}_s{args.shard}"; fn = f"followup_mk/results_e58b_{args.ps}{sfx}.jsonl"
    done = set()
    if os.path.exists(fn):
        for l in open(fn):
            try: done.add(json.loads(l)["row"])
            except Exception: pass
    if args.ps == "base":
        mine_rows = [i for _, _, ix in mine for i in ix]; widx = {i: q for q, i in enumerate(mine_rows)}
        hp = f"{OUTD}/hid{sfx}.npy"
        H = np.lib.format.open_memmap(hp, mode="r+" if os.path.exists(hp) else "w+", dtype=np.float16, shape=(len(mine_rows), 2, 4096))
        json.dump(mine_rows, open(f"{OUTD}/hid{sfx}_rows.json", "w"))
    else:
        Z = np.load(f"{OUTD}/dirs.npz"); DIR = Z["dirs"]; SCALE = float(Z["scale"])
        BASE = {}
        for l in open(f"followup_mk/results_e58b_base{sfx}.jsonl"): o = json.loads(l); BASE[o["row"]] = o["answer"]
    fo = open(fn, "a", encoding="utf-8"); t0 = time.time(); nb = 0; rep = [0, 0]
    print(f"{args.ps} shard {args.shard}/{args.nshard}: 묶음 {len(mine)}, 이미 {len(done)} 행", flush=True)
    for j, L, ix in mine:
        if all(i in done for i in ix): continue
        b = [pids[i] for i in ix]
        if args.ps == "base":
            a, rec = oom_retry(lambda: generate(m, b, capture=True))
            for q, i in enumerate(ix):
                H[widx[i], 0] = rec["hB"][q].astype(np.float16); H[widx[i], 1] = rec["hK"][q].astype(np.float16)
            H.flush()
            for q, i in enumerate(ix):
                sc = float(np.mean([rec["inter"][q][s] != rec["inter"][q][31] for s in range(31)]))
                fo.write(json.dumps(dict(row=i, batch=j, answer=a[q], type=grade(i, a[q]), lp0=round(rec["lp0"][q], 4), lp4=round(rec["lp4"][q], 4), margin0=round(rec["margin0"][q], 4),
                                         sc=round(sc, 4), final_eq_step31=rec["inter"][q][31] == a[q]), ensure_ascii=False) + "\n")
        else:
            vec = torch.tensor(np.stack([DIR[grp[i]] * SCALE for i in ix]), device="cuda", dtype=torch.float32)
            if nb < 2:
                ab, _ = oom_retry(lambda: generate(m, b)); rep[0] += sum(x_ == BASE[i] for x_, i in zip(ab, ix)); rep[1] += len(ix)
            a, _ = oom_retry(lambda: generate(m, b, add=vec))
            for q, i in enumerate(ix): fo.write(json.dumps(dict(row=i, batch=j, answer=a[q], type=grade(i, a[q])), ensure_ascii=False) + "\n")
        fo.flush(); nb += 1
        if nb % 20 == 0: print(f"  {nb}/{len(mine)} 묶음, {time.time() - t0:.0f}s", flush=True)
    if args.ps == "steer": print(f"재현 (base 다시 돌림, 첫 2 묶음): {rep[0]}/{rep[1]} 같은 답", flush=True)
    print(f"끝: {time.time() - t0:.0f}s", flush=True); print("DONE", flush=True)


if __name__ == "__main__": main()
