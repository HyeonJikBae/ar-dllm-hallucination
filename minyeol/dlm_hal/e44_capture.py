"""E44 (PREREG.md): 첫 칸 결정의 촘촘한 궤적 수집.  개입 없는 생성 (topk_margin, 32/32, 0-shot).
행: e12_regen 과 같은 R (1~4/5 5,950 + E7 5/5 2,000, 같은 순서·같은 by_length 묶음) + E7 0/5 앞 500 (따로 묶음).
조각 (shard) 마다 따로 디렉터리 followup_mk/e44/shard{s}/ 에 저장 (NFS 동시 쓰기 피함).  묶음 번호 % nshards == s 인 묶음만 처리.  재개 가능.
usage: python followup_mk/e44_capture.py --shard S --nshards N [--limit 32]"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
SH = int(sys.argv[sys.argv.index("--shard") + 1]) if "--shard" in sys.argv else 0
NS = int(sys.argv[sys.argv.index("--nshards") + 1]) if "--nshards" in sys.argv else 1
LAY = [12, 16, 20, 24, 27]; SV = list(range(9)) + [16, 31]; TOPK = 10
_, Z7, F7 = e7_rows()
R = [int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]; Z = [int(i) for i in Z7[:500]]
setof = {**{i: ("mid" if mid[i] else "five") for i in R}, **{i: "zero" for i in Z}}
m = Dream(); tok = m.tok
ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R + Z}
batches = [("R", L, ix) for L, ix in by_length(R, lambda i: ids[i])] + [("Z", L, ix) for L, ix in by_length(Z, lambda i: ids[i])]
if LIM:   # 파일럿: 앞쪽 묶음 몇 개 (R 과 Z 를 섞어)
    batches = batches[:max(1, LIM // BS)] + [b for b in batches if b[0] == "Z"][:1]
mine = [b for j, b in enumerate(batches) if j % NS == SH]
rows_mine = [i for _, _, ix in mine for i in ix]; n = len(rows_mine)
D = f"{OUT}/e44{'_pilot' if LIM else ''}/shard{SH}"; os.makedirs(D, exist_ok=True)
np.save(f"{D}/rows.npy", np.array(rows_mine, dtype=np.int64))
json.dump(dict(layers=LAY, steps=SV, position="prompt_end (L-1) = 첫 칸 예측 자리", topk=TOPK, shard=SH, nshards=NS, canvas=CANVAS, n=n,
               gold_rule="tok(' ' + gold_value)[0]"), open(f"{D}/config.json", "w"), ensure_ascii=False)
mm = lambda name, shape, dt: np.lib.format.open_memmap(f"{D}/{name}.npy", mode="r+" if os.path.exists(f"{D}/{name}.npy") else "w+", dtype=dt, shape=shape)
HID = mm("hid", (n, len(SV), len(LAY), 3584), np.float16)
LGR = mm("ll_gold_rank", (n, len(SV), len(LAY)), np.int32); LGL = mm("ll_gold_logit", (n, len(SV), len(LAY)), np.float32)
LTI = mm("ll_top_ids", (n, len(SV), len(LAY), TOPK), np.int32); LTL = mm("ll_top_logit", (n, len(SV), len(LAY), TOPK), np.float32)
COM = mm("committed", (n, STEPS), np.int16); S0M = mm("slot0_mask", (n, STEPS), bool); S0A = mm("slot0_argmax", (n, STEPS), np.int32)
S0R = mm("slot0_gold_rank", (n, STEPS), np.int32); S0G = mm("slot0_gold_logit", (n, STEPS), np.float32); S0T = mm("slot0_top1_logit", (n, STEPS), np.float32)
done = np.load(f"{D}/done.npy") if os.path.exists(f"{D}/done.npy") else np.zeros(n, bool)
where = {i: j for j, i in enumerate(rows_mine)}
gold = {i: tok(" " + rows[i]["gold_value"], add_special_tokens=False)["input_ids"][0] for i in rows_mine}
norm_f = m.model.model.norm; head = m.model.lm_head
meta = open(f"{D}/meta.jsonl", "a", encoding="utf-8"); t0 = time.time(); cnt = 0
print(f"shard {SH}/{NS}: 묶음 {len(mine)}, 행 {n}, 이미 {int(done.sum())}", flush=True)
for tag, L, ix in mine:
    J = np.array([where[i] for i in ix])
    if done[J].all(): continue
    nb = len(ix); g = torch.tensor([gold[i] for i in ix], device="cuda"); ar = torch.arange(nb, device="cuda")
    hb = torch.zeros(nb, len(SV), len(LAY), 3584, dtype=torch.float16, device="cuda")
    gr = torch.zeros(nb, len(SV), len(LAY), dtype=torch.int32, device="cuda"); gl = torch.zeros(nb, len(SV), len(LAY), device="cuda")
    ti = torch.zeros(nb, len(SV), len(LAY), TOPK, dtype=torch.int32, device="cuda"); tl = torch.zeros(nb, len(SV), len(LAY), TOPK, device="cuda")
    def mk(li):
        def f(h, step):
            if step in SV:
                si = SV.index(step); x = h[:, L - 1]; hb[:, si, li] = x.to(torch.float16)
                z = head(norm_f(x.unsqueeze(1))).squeeze(1).float()       # logit lens: 첫 칸 로짓
                gv = z[ar, g]; gl[:, si, li] = gv; gr[:, si, li] = (z > gv[:, None]).sum(-1).int()
                tv = z.topk(TOPK, -1); ti[:, si, li] = tv.indices.int(); tl[:, si, li] = tv.values
        return f
    com = torch.zeros(nb, STEPS, dtype=torch.int16, device="cuda"); s0m = torch.zeros(nb, STEPS, dtype=torch.bool, device="cuda"); s0a = torch.zeros(nb, STEPS, dtype=torch.int32, device="cuda")
    s0r = torch.zeros(nb, STEPS, dtype=torch.int32, device="cuda"); s0g = torch.zeros(nb, STEPS, device="cuda"); s0t = torch.zeros(nb, STEPS, device="cuda")
    def lh(step, x, logits):
        cv = x[:, -CANVAS:] == m.mask_id; com[:, step] = (CANVAS - cv.sum(-1)).to(torch.int16); s0m[:, step] = cv[:, 0]
        z = logits[:, -CANVAS].float(); s0a[:, step] = torch.softmax(logits[:, -CANVAS], -1).max(-1).indices.int()   # Dream 의 bf16 softmax 동점 처리와 같게
        gv = z[ar, g]; s0g[:, step] = gv; s0r[:, step] = (z > gv[:, None]).sum(-1).int(); s0t[:, step] = z.max(-1).values
        return logits
    ans = m.generate([ids[i] for i in ix], {l_: mk(li) for li, l_ in enumerate(LAY)}, logits_hook=lh)
    HID[J] = hb.cpu().numpy(); LGR[J] = gr.cpu().numpy(); LGL[J] = gl.cpu().numpy(); LTI[J] = ti.cpu().numpy(); LTL[J] = tl.cpu().numpy()
    COM[J] = com.cpu().numpy(); S0M[J] = s0m.cpu().numpy(); S0A[J] = s0a.cpu().numpy(); S0R[J] = s0r.cpu().numpy(); S0G[J] = s0g.cpu().numpy(); S0T[J] = s0t.cpu().numpy()
    s0m_ = s0m.cpu().numpy(); s0a_ = s0a.cpu().numpy()
    for j, (i, a) in enumerate(zip(ix, ans)):
        lm_ = np.where(s0m_[j])[0]; last = int(lm_.max()) if len(lm_) else -1
        meta.write(json.dumps({"row": i, "set": setof[i], "k": int(k[i]), "batch": tag, "answer": a, "new_type": m.grade(i, a), "gold_id": int(gold[i]),
                               "slot0_last_masked_step": last, "slot0_final_token": int(s0a_[j, last]) if last >= 0 else None}, ensure_ascii=False) + "\n")
    done[J] = True; cnt += nb; meta.flush()
    if cnt % 320 < nb:
        for a_ in (HID, LGR, LGL, LTI, LTL, COM, S0M, S0A, S0R, S0G, S0T): a_.flush()
        np.save(f"{D}/done.npy", done); print(f"  {int(done.sum())}/{n} {time.time() - t0:.0f}s", flush=True)
for a_ in (HID, LGR, LGL, LTI, LTL, COM, S0M, S0A, S0R, S0G, S0T): a_.flush()
np.save(f"{D}/done.npy", done); print("DONE", flush=True)
