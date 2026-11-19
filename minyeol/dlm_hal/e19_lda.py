"""E19 (PREREG.md): E17 LDA 방향 조향, 행 S.  usage: python followup_mk/e19_lda.py --conds "base,LDA_mid_0.9 c1" --tag A [--limit N]"""
import collections, json, os, re, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
CN = sys.argv[sys.argv.index("--conds") + 1].split(","); TAG = sys.argv[sys.argv.index("--tag") + 1]
DIRS = np.load(f"{OUT}/e17_dirs.npz"); _, norm = load_X()
S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}; R = [i for _, i in S]
SPEC = {}
for c in CN:
    if c == "base": SPEC[c] = None; continue
    mm = re.fullmatch(r"(\S+) c([0-9.]+)", c); assert mm and mm.group(1) in DIRS.files, c; SPEC[c] = (mm.group(1), float(mm.group(2)))
fout = f"{OUT}/results_e19{'_pilot' if LIM else ''}_{TAG}.jsonl"; have = collections.defaultdict(set)
if os.path.exists(fout):
    for l in open(fout, encoding="utf-8"):
        try: d = json.loads(l); have[d["cond"]].add(d["row"])
        except Exception: pass
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R}; fo = open(fout, "a", encoding="utf-8")
print(f"행 {len(R)}, 조건 {CN}, norm {norm:.1f}", flush=True)
for cname in CN:
    t0 = time.time()
    for L, ix in by_length(R, lambda i: ids[i]):
        if all(i in have[cname] for i in ix): continue
        fns = {}
        if SPEC[cname] is not None:
            key, c = SPEC[cname]; V = np.stack([DIRS[key][grp[i]] for i in ix]); V = V / np.linalg.norm(V, axis=1, keepdims=True)
            V = torch.from_numpy(V).float().cuda()
            def f(h, step, V=V, L=L, c=c): h[:, L - 1] = (h[:, L - 1].float() + V * c * norm).to(h.dtype)
            fns[LAYER] = f
        ans = m.generate([ids[i] for i in ix], fns)
        for i, a_ in zip(ix, ans):
            if i in have[cname]: continue
            fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "k": int(k[i]), "answer": a_, "new_type": m.grade(i, a_)}, ensure_ascii=False) + "\n"); have[cname].add(i)
        fo.flush()
    print(f"  {cname}: 끝 {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
