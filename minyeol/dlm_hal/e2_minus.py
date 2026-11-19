"""E2 (PREREG.md): 정답 행에 -B, B 성분 지우기, +B, 무작위 대조.  usage: python followup_mk/e2_minus.py [--limit N]"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
X, norm = load_X(); dirs = directions(X)
tgt = np.random.default_rng(7).permutation(np.where(mid & (ty == "correct"))[0])[:600]
if LIM: tgt = tgt[:LIM]
hb = np.array([X[i] @ dirs[grp[i]]["B"] for i in tgt]); print(f"정답 행 h.B: 평균 {hb.mean():.2f}, 표준편차 {hb.std():.2f}, |h| 평균 {np.linalg.norm(X[tgt], axis=1).mean():.1f}, c=1 크기 {norm:.1f}", flush=True)
G = {f"G{q}": unit(np.random.default_rng(1000 + q).standard_normal(3584)) for q in range(4)}
vec = lambda i, d: dirs[grp[i]][d] if d in dirs[grp[i]] else G[d]
CONDS = [("base", None, 0.)] + [(f"-B c{c}", "B", -c) for c in (.5, 1, 2)] + [("projB", "B", None)] + [(f"+B c{c}", "B", c) for c in (1, 2)] + [(f"{d} c1", d, 1.) for d in ("G0", "G1", "G2", "G3", "D0", "D1", "D2", "D3")]
m = Dream(); ids = {int(i): m.ids(build_prompt(rows[i]["question"])) for i in tgt}
fo = open(f"{OUT}/results_e2{'_pilot' if LIM else ''}.jsonl", "w", encoding="utf-8")
for cname, d, c in CONDS:
    t0 = time.time()
    for L, ix in by_length([int(i) for i in tgt], lambda i: ids[i]):
        fns = {}
        if d is not None:
            V = torch.from_numpy(np.stack([vec(i, d) for i in ix])).float().cuda()
            if c is None:
                def f(h, step, V=V, L=L):
                    x = h[:, L - 1].float(); h[:, L - 1] = (x - (x * V).sum(-1, keepdim=True) * V).to(h.dtype)
                fns[LAYER] = f
            else: fns[LAYER] = add_end(V * c * norm, L)
        ans = m.generate([ids[i] for i in ix], fns)
        for i, a in zip(ix, ans): fo.write(json.dumps({"cond": cname, "row": i, "k": int(k[i]), "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n")
    fo.flush(); print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
