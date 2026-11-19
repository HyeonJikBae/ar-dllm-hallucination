"""E57 (b) 보조 (GPU): Qwen2.5-7B 한 번 forward 로 같은 프롬프트의 답 첫 자리 분포를 읽는다.
행 = E7 행 (1~4/5 5,950 + 0/5 2,000 + 5/5 2,000).  금 첫 토큰 = tok(' ' + gold_value)[0] (E44 gold_rule 과 같음).
저장 (followup_mk/e57_qwen/): rows, gold_id, gold_lp (log softmax), gold_rank (0 = 1 위), top1_id, top1_lp, margin (1·2 위 로짓 차), done.
같은 길이 프롬프트끼리 묶어 패딩 없이 forward.  usage: python followup_mk/e57_qwen_gold.py [--pilot]"""
import json, os, sys, time, collections
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import rows, build_prompt, e7_rows, OUT
from transformers import AutoTokenizer, AutoModelForCausalLM

PILOT = "--pilot" in sys.argv
D = f"{OUT}/e57_qwen" + ("_pilot" if PILOT else ""); os.makedirs(D, exist_ok=True)
R, _, _ = e7_rows(); R = R[:64] if PILOT else R; n = len(R)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.bfloat16).cuda().eval()
f = lambda name, dt, fill: np.load(f"{D}/{name}.npy") if os.path.exists(f"{D}/{name}.npy") else np.full(n, fill, dt)
A = {"gold_id": f("gold_id", np.int64, -1), "gold_lp": f("gold_lp", np.float32, np.nan), "gold_rank": f("gold_rank", np.int64, -1),
     "top1_id": f("top1_id", np.int64, -1), "top1_lp": f("top1_lp", np.float32, np.nan), "margin": f("margin", np.float32, np.nan)}
done = f("done", bool, False); np.save(f"{D}/rows.npy", R)
ids = {j: tok(build_prompt(rows[int(r)]["question"]))["input_ids"] for j, r in enumerate(R) if not done[j]}
byL = collections.defaultdict(list)
for j, x in ids.items(): byL[len(x)].append(j)
t0 = time.time(); nb = 0
with torch.inference_mode():
    for L, js in sorted(byL.items()):
        for b0 in range(0, len(js), 32):
            jb = js[b0:b0 + 32]; inp = torch.tensor([ids[j] for j in jb], device="cuda")
            lg = model(input_ids=inp).logits[:, -1].float(); lp = torch.log_softmax(lg, -1)
            top2 = lg.topk(2, -1)
            for q, j in enumerate(jb):
                g = tok(" " + rows[int(R[j])]["gold_value"])["input_ids"][0]
                A["gold_id"][j] = g; A["gold_lp"][j] = float(lp[q, g]); A["gold_rank"][j] = int((lg[q] > lg[q, g]).sum())
                A["top1_id"][j] = int(top2.indices[q, 0]); A["top1_lp"][j] = float(lp[q, top2.indices[q, 0]]); A["margin"][j] = float(top2.values[q, 0] - top2.values[q, 1])
                done[j] = True
            nb += 1
            if nb % 50 == 0:
                for kx, v in A.items(): np.save(f"{D}/{kx}.npy", v)
                np.save(f"{D}/done.npy", done); print(f"{done.sum()}/{n} {time.time() - t0:.0f}s", flush=True)
for kx, v in A.items(): np.save(f"{D}/{kx}.npy", v)
np.save(f"{D}/done.npy", done); print(f"DONE {done.sum()}/{n} {time.time() - t0:.0f}s", flush=True)
