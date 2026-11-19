"""E37 (PREREG.md): 0/5 에 금 첫 토큰 고정.  usage: python followup_mk/e37_zero_first.py"""
import collections, json, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import dloop_p2 as dl
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r"))
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]; tok0 = {int(r): int(committed[j, 0]) for j, r in enumerate(R7)}
m = Dream(); tk = m.tok
def golds(i): r = rows[i]; return [r["gold_value"]] + [a for a in (r.get("gold_aliases") or []) if a]
rules = {"공백": lambda s: tk(" " + s, add_special_tokens=False)["input_ids"][0], "없음": lambda s: tk(s, add_special_tokens=False)["input_ids"][0]}
cr = [int(r) for r in R7 if int(r) in meta and meta[int(r)]["new_type"] == "correct"][:3000]
score = {n: np.mean([tok0[i] in {f(g) for g in golds(i) if g} for i in cr]) for n, f in rules.items()}
best = max(score, key=score.get); rule = rules[best]; print("형식 규칙 일치율", {k_: round(float(v), 3) for k_, v in score.items()}, "->", best, flush=True)
def gtok(i): return rule(rows[i]["gold_value"])
S = set_S(); st = {i: s for s, i in S}; rel = {i: rows[i]["relation"] for _, i in S}
zero = [i for s, i in S if s == "zero"]; wrong = [i for s, i in S if s in ("broken", "entity")]
byrel = collections.defaultdict(list)
for i in zero: byrel[rel[i]].append(i)
rnd = {}
for i in zero:
    pool = [j for j in byrel[rel[i]] if j != i and gtok(j) != gtok(i)]
    if pool: rnd[i] = gtok(pool[np.random.default_rng(i).integers(len(pool))])
conds = [("base", zero, lambda i: -1), ("금토큰고정", zero, gtok), ("무작위금토큰", [i for i in zero if i in rnd], lambda i: rnd[i]), ("검증: 오답행 금토큰", wrong, gtok)]
ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}; fo = open(f"{OUT}/results_e37.jsonl", "w", encoding="utf-8")
for cname, rs, f in conds:
    t0 = time.time()
    for L, ix in by_length(rs, lambda i: ids[i]):
        pf = [f(i) for i in ix]; ans, _ = dl.generate(m, [ids[i] for i in ix], prefill=(None if all(p < 0 for p in pf) else pf))
        for i, a in zip(ix, ans): fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n")
    fo.flush(); print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
R = collections.defaultdict(list)
for l in open(f"{OUT}/results_e37.jsonl"): d = json.loads(l); R[d["cond"]].append(d["new_type"] == "correct")
print({c: (len(v), round(100 * float(np.mean(v)), 1)) for c, v in R.items()}); print("DONE", flush=True)
