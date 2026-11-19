"""E6 (PREREG.md): 5/5, 0/5 상태로 만든 '아는 상태' 방향 K 로 조향.  usage: python followup_mk/e6_know.py [--limit N]"""
import collections, json, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
X, norm = load_X(); dirs = directions(X); rel = np.array([r["relation"] for r in rows])
def relmean(a_mask, b_mask, tr, minn):
    ds = []
    for r_ in np.unique(rel):
        a = np.where(tr & a_mask & (rel == r_))[0]; b = np.where(tr & b_mask & (rel == r_))[0]
        if len(a) >= minn and len(b) >= minn: ds.append(X[a].mean(0) - X[b].mean(0))
    return unit(np.mean(ds, 0)), len(ds)
for g in range(NF):
    tr = grp != g
    dirs[g]["K"], nk = relmean(k == 5, k == 0, tr, 20)
    dirs[g]["K_pool"] = unit(X[tr & (k == 5)].mean(0) - X[tr & (k == 0)].mean(0))
    dirs[g]["KC"], nkc = relmean((k == 5) & (ty == "correct"), (k == 0) & (ty == "broken"), tr, 10)
    B, K = dirs[g]["B"], dirs[g]["K"]
    dirs[g]["K⊥B"] = unit(K - (K @ B) * B); dirs[g]["B⊥K"] = unit(B - (B @ K) * K)
    print(f"fold {g}: K 관계 {nk}, KC 관계 {nkc}, cos(K,B) {K @ B:.3f}, cos(K_pool,B) {dirs[g]['K_pool'] @ B:.3f}, cos(KC,B) {dirs[g]['KC'] @ B:.3f}, cos(K,K_pool) {K @ dirs[g]['K_pool']:.3f}", flush=True)
for q in range(2):
    v = unit(np.random.default_rng(1000 + q).standard_normal(3584))
    for g in range(NF): dirs[g][f"G{q}"] = v
r5 = np.random.default_rng(5)
SETS = {"broken": np.where(mid & (ty == "broken"))[0], "entity": r5.permutation(np.where(mid & (ty == "entity"))[0])[:300], "correct": r5.permutation(np.where(mid & (ty == "correct"))[0])[:300],
        "zero": np.random.default_rng(9).permutation(np.where(k == 0)[0])[:300]}
tgt = [(s, int(i)) for s, v in SETS.items() for i in (v[:LIM] if LIM else v)]; st = {i: s for s, i in tgt}
CONDS = [("base", None, 0.), ("B c1", "B", 1.)] + [(f"K c{c}", "K", c) for c in (.5, 1., 2.)] + [(f"{d} c1", d, 1.) for d in ("K_pool", "KC", "K⊥B", "B⊥K", "G0", "G1")]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in tgt}
fo = open(f"{OUT}/results_e6{'_pilot' if LIM else ''}.jsonl", "w", encoding="utf-8")
for cname, d, c in CONDS:
    t0 = time.time()
    for L, ix in by_length([i for _, i in tgt], lambda i: ids[i]):
        fns = {} if d is None else {LAYER: add_end(torch.from_numpy(np.stack([dirs[grp[i]][d] for i in ix])).float().cuda() * c * norm, L)}
        for i, a in zip(ix, m.generate([ids[i] for i in ix], fns)): fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "k": int(k[i]), "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n")
    fo.flush(); print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
