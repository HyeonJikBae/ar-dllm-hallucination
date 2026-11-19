"""E33 (PREREG.md): 전체 규모 3-shot.  행 = E12 와 같은 1~4/5 5,950 + E7 5/5 2,000.  프롬프트 구성은 e3_fewshot.py 와 같음 (INSTRUCTION + 예시 3 개 + 질문).
예시는 겹 g 마다 다른 겹 (grp != g) 의 5/5 'question' 판 사실에서 관계가 서로 다른 3 개 (gold 30 자 이하, seed 11).  usage: python followup_mk/e33_fewshot_full.py [--limit N] [--tag T]"""
import collections, json, os, sys, time
import numpy as np
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "A"
_, _, F7 = e7_rows(); R = [int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]
st = {i: ("mid" if mid[i] else "five") for i in R}
if LIM: R = [x for j, x in enumerate(R) if j % max(1, len(R) // LIM) == 0][:LIM]
EX = {}
for g in range(NF):
    ex = []; seen = set()
    for i in np.random.default_rng(11).permutation(np.where((k == 5) & (grp != g) & np.array([r["version"] == "question" for r in rows]))[0]):
        r = rows[i]
        if r["relation"] in seen or len(r["gold_value"]) > 30: continue
        ex.append((r["question"], r["gold_value"], r["relation"], int(i))); seen.add(r["relation"])
        if len(ex) == 3: break
    EX[g] = ex
FEW = {g: INSTRUCTION_PROMPT + "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a, _, _ in EX[g]) for g in range(NF)}
exf = f"{OUT}/e33_examples.json"
if not os.path.exists(exf): json.dump({str(g): EX[g] for g in EX}, open(exf, "w"), ensure_ascii=False, indent=1)
fout = f"{OUT}/results_e33{'_pilot' if LIM else ''}_{TAG}.jsonl"; have = set()
if os.path.exists(fout):
    for l in open(fout, encoding="utf-8"):
        try: have.add(json.loads(l)["row"])
        except Exception: pass
m = Dream(); ids = {i: m.ids(FEW[grp[i]] + f"Question: {rows[i]['question']}\nAnswer:") for i in R}
print(f"행 {len(R)}, 이미 {len(have)}", flush=True); fo = open(fout, "a", encoding="utf-8"); t0 = time.time(); nb = 0
for L, ix in by_length(R, lambda i: ids[i]):
    if all(i in have for i in ix): continue
    for i, a in zip(ix, m.generate([ids[i] for i in ix])):
        if i in have: continue
        fo.write(json.dumps({"cond": "3shot", "set": st[i], "row": i, "k": int(k[i]), "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n"); have.add(i)
    fo.flush(); nb += 1
    if nb % 50 == 0: print(f"  {len(have)}/{len(R)} {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
