"""E9 (PREREG.md): 조향 방식 (step 구간, 목표 투영, 관계별 방향).  행 S (1,510).  usage: python followup_mk/e9_steer.py [--limit N]"""
import collections, json, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
X, norm = load_X(); dirs = directions(X); rel = np.array([r["relation"] for r in rows])
S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}
TAU = {}; BREL = {}
for g in range(NF):
    tr = grp != g; hb = X[tr & mid & (ty == "correct")] @ dirs[g]["B"]; TAU[g] = {p: float(np.percentile(hb, p)) for p in (50, 75, 90)}
    byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & (ty != "abstain"))[0]: byf[fid[i]]["c" if ty[i] == "correct" else "w"].append(i)
    per = collections.defaultdict(list)
    for f_, v in byf.items():
        if v["c"] and v["w"]: per[rows[v["c"][0]]["relation"]].append(X[v["c"]].mean(0) - X[v["w"]].mean(0))
    BREL[g] = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}
    print(f"fold {g}: tau {TAU[g]}, 관계별 방향 {len(BREL[g])}개, B 와 cos 평균 {np.mean([v @ dirs[g]['B'] for v in BREL[g].values()]):.2f}", flush=True)
def vecs(ix, kind):
    if kind == "Brel": return np.stack([BREL[grp[i]].get(rel[i], dirs[grp[i]]["B"]) for i in ix])
    return np.stack([dirs[grp[i]]["B"] for i in ix])
CONDS = [("base", None), ("B c1", ("add", "B", 1., 0, 32)), ("B c0.5", ("add", "B", .5, 0, 32)), ("B c1 s0-3", ("add", "B", 1., 0, 4)), ("B c1 s0-7", ("add", "B", 1., 0, 8)),
         ("B c1 s8-31", ("add", "B", 1., 8, 32))] + [(f"ACE p{p}", ("ace", "B", p, 0, 32)) for p in (50, 75, 90)] + [(f"ACE t{t}", ("ace_abs", "B", t, 0, 32)) for t in (27, 54)] + [("Brel c1", ("add", "Brel", 1., 0, 32))]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}
fo = open(f"{OUT}/results_e9{'_pilot' if LIM else ''}.jsonl", "w", encoding="utf-8")
for cname, spec in CONDS:
    t0 = time.time()
    for L, ix in by_length([i for _, i in S], lambda i: ids[i]):
        fns = {}
        if spec is not None:
            op, kind, c, a, b = spec; V = torch.from_numpy(vecs(ix, kind)).float().cuda()
            if op == "add":
                def f(h, step, V=V, L=L, c=c, a=a, b=b):
                    if a <= step < b: h[:, L - 1] = (h[:, L - 1].float() + V * c * norm).to(h.dtype)
            else:
                T = torch.tensor([TAU[grp[i]][c] if op == "ace" else float(c) for i in ix], device="cuda")[:, None]
                def f(h, step, V=V, L=L, T=T):
                    x = h[:, L - 1].float(); pr = (x * V).sum(-1, keepdim=True); h[:, L - 1] = (x + torch.clamp(T - pr, min=0) * V).to(h.dtype)
            fns[LAYER] = f
        for i, a_ in zip(ix, m.generate([ids[i] for i in ix], fns)): fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "k": int(k[i]), "answer": a_, "new_type": m.grade(i, a_)}, ensure_ascii=False) + "\n")
    fo.flush(); print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
