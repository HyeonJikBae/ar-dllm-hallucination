"""E3 (PREREG.md): zero-shot 대 3-shot.  대상은 caa_directions.py 와 같은 행.  usage: python followup_mk/e3_fewshot.py [--limit N]"""
import json, sys, time
import numpy as np
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
r5 = np.random.default_rng(5)
SETS = {"broken": np.where(mid & (ty == "broken"))[0], "entity": r5.permutation(np.where(mid & (ty == "entity"))[0])[:300], "correct": r5.permutation(np.where(mid & (ty == "correct"))[0])[:300]}
tgt = [(s, int(i)) for s, v in SETS.items() for i in (v[:LIM] if LIM else v)]
used = {fid[i] for _, i in tgt}; ex = []; seen = set()
for i in np.random.default_rng(11).permutation(np.where((k == 5) & np.array([r["version"] == "question" for r in rows]))[0]):
    r = rows[i]
    if r["relation"] in seen or fid[i] in used or len(r["gold_value"]) > 30: continue
    ex.append((r["question"], r["gold_value"], r["relation"])); seen.add(r["relation"])
    if len(ex) == 3: break
print("예시:", ex, flush=True)
FEW = INSTRUCTION_PROMPT + "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a, _ in ex)
PROMPTS = {"zero": lambda q: build_prompt(q), "3shot": lambda q: FEW + f"Question: {q}\nAnswer:"}
m = Dream(); fo = open(f"{OUT}/results_e3{'_pilot' if LIM else ''}.jsonl", "w", encoding="utf-8")
json.dump(dict(examples=ex), open(f"{OUT}/e3_examples.json", "w"), ensure_ascii=False)
for cname, pf in PROMPTS.items():
    t0 = time.time(); ids = {i: m.ids(pf(rows[i]["question"])) for _, i in tgt}; st = dict((i, s) for s, i in tgt)
    for L, ix in by_length([i for _, i in tgt], lambda i: ids[i]):
        for i, a in zip(ix, m.generate([ids[i] for i in ix])): fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "k": int(k[i]), "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n")
    fo.flush(); print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
