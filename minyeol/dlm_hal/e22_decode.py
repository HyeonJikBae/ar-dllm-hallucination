"""E22 (PREREG.md): 확정 순서 통제.  행 S (1,510).  조건: base (orig) | ar (왼쪽에서 한 칸씩) | [사후] ar + Brel c1 | [사후] orig + Brel c1.
usage: python followup_mk/e22_decode.py [--limit N]   (daissv4 에서 실행)"""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import dloop_p4 as dloop
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
X, norm = load_X(); dirs = directions(X); rel = np.array([r["relation"] for r in rows])
S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}
BREL = {}
for g in range(NF):   # e9_steer.py / e12_regen.py 와 같은 관계별 방향
    tr = grp != g; byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & (ty != "abstain"))[0]: byf[fid[i]]["c" if ty[i] == "correct" else "w"].append(i)
    per = collections.defaultdict(list)
    for f_, v in byf.items():
        if v["c"] and v["w"]: per[rows[v["c"][0]]["relation"]].append(X[v["c"]].mean(0) - X[v["w"]].mean(0))
    BREL[g] = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}
def vecs(ix): return np.stack([BREL[grp[i]].get(rel[i], dirs[grp[i]]["B"]) for i in ix])
CONDS = [("base", "orig", False), ("ar", "ar", False), ("ar + Brel c1 [사후]", "ar", True), ("orig + Brel c1 [사후]", "orig", True)]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}
fout = f"{OUT}/results_e22{'_pilot' if LIM else ''}.jsonl"; have = collections.defaultdict(set)
if os.path.exists(fout):
    for l in open(fout, encoding="utf-8"):
        try: d = json.loads(l); have[d["cond"]].add(d["row"])
        except Exception: pass
fo = open(fout, "a", encoding="utf-8")
for cname, mode, steer in CONDS:
    t0 = time.time()
    for L, ix in by_length([i for _, i in S], lambda i: ids[i]):
        if all(i in have[cname] for i in ix): continue
        fns = {}
        if steer:
            V = torch.from_numpy(vecs(ix)).float().cuda()
            def f(h, step, V=V, L=L): h[:, L - 1] = (h[:, L - 1].float() + V * 1.0 * norm).to(h.dtype)
            fns[LAYER] = f
        ans, _ = dloop.generate(m, [ids[i] for i in ix], fns=fns, mode=mode)
        for i, a in zip(ix, ans):
            if i in have[cname]: continue
            fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "k": int(k[i]), "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n"); have[cname].add(i)
        fo.flush()
    print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
