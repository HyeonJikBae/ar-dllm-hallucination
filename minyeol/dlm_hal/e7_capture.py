"""E7 (PREREG.md): 개입 없는 생성의 궤적과 여러 층·위치·step 상태 수집.  재시작 가능.  usage: python followup_mk/e7_capture.py [--limit N]"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
D7 = f"{OUT}/e7{'_pilot' if LIM else ''}"; os.makedirs(D7, exist_ok=True)
R7, Z7, F7 = e7_rows()
if LIM: R7 = np.concatenate([np.random.default_rng(3).permutation(np.where(mid)[0])[:3 * LIM], Z7[:LIM], F7[:LIM]]).astype(int)
n = len(R7); np.save(f"{D7}/rows.npy", R7)
LAY = [4, 8, 12, 16, 20, 24, 27]; SV = [0, 4, 8]; PS = ["prompt_end", "q_last", "subj_last"]
json.dump(dict(layers=LAY, steps=SV, positions=PS), open(f"{D7}/config.json", "w"))
mm = lambda name, shape, dt: np.lib.format.open_memmap(f"{D7}/{name}.npy", mode="r+" if os.path.exists(f"{D7}/{name}.npy") else "w+", dtype=dt, shape=shape)
HID = mm("hid", (n, len(SV), len(LAY), len(PS), 3584), np.float16); IDS = mm("ids", (n, STEPS, CANVAS), np.int32)
LP = mm("lp", (n, STEPS, CANVAS), np.float16); MG = mm("mg", (n, STEPS, CANVAS), np.float16); MK = mm("mask", (n, STEPS, CANVAS), bool)
done = np.load(f"{D7}/done.npy") if os.path.exists(f"{D7}/done.npy") else np.zeros(n, bool)
POS = {int(i): positions(int(i)) for i in R7}; where = {int(i): j for j, i in enumerate(R7)}
m = Dream(); meta = open(f"{D7}/meta.jsonl", "a", encoding="utf-8"); t0 = time.time(); cnt = 0
for L, ix in by_length([int(i) for i in R7 if not done[where[int(i)]]], lambda i: POS[i][0]):
    nb = len(ix); buf = torch.zeros(nb, len(SV), len(LAY), len(PS), 3584, dtype=torch.float16, device="cuda")
    pidx = torch.tensor([[L - 1, POS[i][1], POS[i][2] if POS[i][2] is not None else L - 1] for i in ix], device="cuda"); ar = torch.arange(nb, device="cuda")[:, None]
    def mk(li):
        def f(h, step):
            if step in SV: buf[:, SV.index(step), li] = h[ar, pidx].to(torch.float16)
        return f
    ids_ = torch.zeros(nb, STEPS, CANVAS, dtype=torch.int32, device="cuda"); lp_ = torch.zeros(nb, STEPS, CANVAS, device="cuda"); mg_ = torch.zeros_like(lp_); mk_ = torch.zeros(nb, STEPS, CANVAS, dtype=torch.bool, device="cuda")
    def lh(step, x, logits):
        z = logits[:, -CANVAS:].float(); t2 = z.topk(2, -1); lse = torch.logsumexp(z, -1)
        ids_[:, step] = torch.softmax(logits[:, -CANVAS:], -1).max(-1).indices.int(); lp_[:, step] = t2.values[..., 0] - lse; mg_[:, step] = t2.values[..., 0] - t2.values[..., 1]; mk_[:, step] = x[:, -CANVAS:] == m.mask_id
        return logits
    ans = m.generate([POS[i][0] for i in ix], {l_: mk(li) for li, l_ in enumerate(LAY)}, logits_hook=lh)
    J = np.array([where[i] for i in ix]); HID[J] = buf.cpu().numpy(); IDS[J] = ids_.cpu().numpy(); LP[J] = lp_.cpu().numpy().astype(np.float16); MG[J] = mg_.cpu().numpy().astype(np.float16); MK[J] = mk_.cpu().numpy()
    for i, a in zip(ix, ans): meta.write(json.dumps({"row": i, "k": int(k[i]), "subj_found": POS[i][2] is not None, "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n")
    done[J] = True; cnt += nb; meta.flush()
    if cnt % 320 < nb:
        for a_ in (HID, IDS, LP, MG, MK): a_.flush()
        np.save(f"{D7}/done.npy", done); print(f"  {cnt}/{n - 0} {time.time() - t0:.0f}s", flush=True)
for a_ in (HID, IDS, LP, MG, MK): a_.flush()
np.save(f"{D7}/done.npy", done); print("DONE", flush=True)
