"""E34 (PREREG.md): OSCAR 식 감지 기준선.  행 S (1,510).  alg origin (무작위 확정 순서), 체인 c 마다 묶음 b 의 시드 = 7919*c + 13*b + 1.
답 칸 0~3 의 원 캔버스 토큰과 최종 답을 저장.  usage: python followup_mk/e34_oscar.py --chains 0,1,2,3 [--limit N] [--tag T]"""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "A"
CH = [int(c) for c in sys.argv[sys.argv.index("--chains") + 1].split(",")]
S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}; R = [i for _, i in S]
fout = f"{OUT}/results_e34{'_pilot' if LIM else ''}_{TAG}.jsonl"; have = set()
if os.path.exists(fout):
    for l in open(fout, encoding="utf-8"):
        try: d = json.loads(l); have.add((d["chain"], d["row"]))
        except Exception: pass
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R}; fo = open(fout, "a", encoding="utf-8")
@torch.inference_mode()
def gen(id_lists, seed):
    torch.manual_seed(seed); inp = torch.tensor(id_lists, device="cuda")
    o = m.model.diffusion_generate(inputs=inp, attention_mask=torch.ones_like(inp), max_new_tokens=CANVAS, steps=STEPS, temperature=0.0, alg="origin", alg_temp=0.0,
                                   mask_token_id=m.mask_id, pad_token_id=m.tok.pad_token_id or m.tok.eos_token_id)
    L = inp.shape[1]; out = []
    for j in range(len(id_lists)):
        full = o[j][L:].tolist(); g_ = full[: next((q for q, t in enumerate(full) if t in m.SP), len(full))]
        out.append((extract_answer(m.tok.decode(g_, skip_special_tokens=True)), full[:4]))
    return out
print(f"행 {len(R)}, 체인 {CH}, 이미 {len(have)}", flush=True)
for c in CH:
    t0 = time.time()
    for b, (L, ix) in enumerate(by_length(R, lambda i: ids[i])):
        if all((c, i) in have for i in ix): continue
        for i, (a, sl) in zip(ix, gen([ids[i] for i in ix], 7919 * c + 13 * b + 1)):
            if (c, i) in have: continue
            fo.write(json.dumps({"chain": c, "set": st[i], "row": i, "k": int(k[i]), "answer": a, "slots": sl, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n"); have.add((c, i))
        fo.flush()
    print(f"  체인 {c}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
