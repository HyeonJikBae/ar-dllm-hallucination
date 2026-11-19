"""E12 (PREREG.md): 전체 규모 다시 생성.  행 = 1/5~4/5 전부 5,950 + E7 의 5/5 2,000.  조건 구현은 e9 (B c0.5, Brel c1, ACE t27) / e8 (maskgit_plus, entropy) 와 같음.
usage: python followup_mk/e12_regen.py [--conds "B c0.5,Brel c1"] [--limit N]   재개 가능: (cond,row) 이미 있으면 건너뜀 (묶음 구성은 유지)."""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
ALL = ["B c0.5", "Brel c1", "ACE t27", "maskgit_plus", "entropy", "base"]   # base: 같은 실행 대조 (민감도 확인용, 주 분석은 E7 이 첫 생성)
CN = sys.argv[sys.argv.index("--conds") + 1].split(",") if "--conds" in sys.argv else ALL
import re as _re
for c in CN:   # E18: "B cX" / "Brel cX" 세기 일반화
    mm = _re.fullmatch(r"(BrelShuf|Brel|Rand|B) c([0-9.]+)", c)
    if c not in ALL and mm: ALL.append(c)
assert all(c in ALL for c in CN), CN
X, norm = load_X(); dirs = directions(X); rel = np.array([r["relation"] for r in rows])
_, _, F7 = e7_rows()
R = [int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]
st = {i: ("mid" if mid[i] else "five") for i in R}
if LIM: R = [x for j, x in enumerate(R) if j % max(1, len(R) // LIM) == 0][:LIM]
BREL = {}
for g in range(NF):   # e9_steer.py 와 같은 관계별 방향
    tr = grp != g; byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & (ty != "abstain"))[0]: byf[fid[i]]["c" if ty[i] == "correct" else "w"].append(i)
    per = collections.defaultdict(list)
    for f_, v in byf.items():
        if v["c"] and v["w"]: per[rows[v["c"][0]]["relation"]].append(X[v["c"]].mean(0) - X[v["w"]].mean(0))
    BREL[g] = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}
def vecs(ix, kind):
    if kind == "Rand": return np.stack([unit(np.random.default_rng(10_000 + int(i)).standard_normal(X.shape[1])) for i in ix])
    if kind == "BrelShuf":   # E28: 그 행의 관계가 아닌 다른 관계의 Brel
        out = []
        for i in ix:
            ks = sorted(BREL[grp[i]]); cur = rel[i]; alt = [r_ for r_ in ks if r_ != cur]; out.append(BREL[grp[i]][alt[int(i) % len(alt)]])
        return np.stack(out)
    if kind == "Brel": return np.stack([BREL[grp[i]].get(rel[i], dirs[grp[i]]["B"]) for i in ix])
    return np.stack([dirs[grp[i]]["B"] for i in ix])
SPEC = {"base": ("none", None), "B c0.5": ("steer", ("add", "B", .5, 0, 32)), "Brel c1": ("steer", ("add", "Brel", 1., 0, 32)), "ACE t27": ("steer", ("ace_abs", "B", 27, 0, 32)),
        "maskgit_plus": ("alg", "maskgit_plus"), "entropy": ("alg", "entropy")}
for c in CN:
    mm = _re.fullmatch(r"(BrelShuf|Brel|Rand|B) c([0-9.]+)", c)
    if mm and c not in SPEC: SPEC[c] = ("steer", ("add", mm.group(1), float(mm.group(2)), 0, 32))
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
fout = f"{OUT}/results_e12{'_pilot' if LIM else ''}{'_' + TAG if TAG else ''}.jsonl"; have = collections.defaultdict(set)   # 파일은 프로세스마다 따로 (동시 append 피함)
if os.path.exists(fout):
    for l in open(fout, encoding="utf-8"):
        try: d = json.loads(l); have[d["cond"]].add(d["row"])
        except Exception: pass
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R}
fo = open(fout, "a", encoding="utf-8")
hv = {c: len(have[c]) for c in CN}
print(f"행 {len(R)} (1~4/5 {sum(st[i] == 'mid' for i in R)}, 5/5 {sum(st[i] == 'five' for i in R)}), 조건 {CN}, 이미 있음 {hv}", flush=True)
for cname in CN:
    t0 = time.time(); kind, spec = SPEC[cname]; nb = 0
    for L, ix in by_length(R, lambda i: ids[i]):
        if all(i in have[cname] for i in ix): continue
        fns = {}; kw = {}
        if kind == "steer":
            op, vk, c, a, b = spec; V = torch.from_numpy(vecs(ix, vk)).float().cuda()
            if op == "add":
                def f(h, step, V=V, L=L, c=c, a=a, b=b):
                    if a <= step < b: h[:, L - 1] = (h[:, L - 1].float() + V * c * norm).to(h.dtype)
            else:
                T = torch.full((len(ix), 1), float(c), device="cuda")
                def f(h, step, V=V, L=L, T=T):
                    x = h[:, L - 1].float(); pr = (x * V).sum(-1, keepdim=True); h[:, L - 1] = (x + torch.clamp(T - pr, min=0) * V).to(h.dtype)
            fns[LAYER] = f
        elif kind == "alg": kw = dict(alg=spec)
        ans = m.generate([ids[i] for i in ix], fns, **kw)
        for i, a_ in zip(ix, ans):
            if i in have[cname]: continue
            fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "k": int(k[i]), "answer": a_, "new_type": m.grade(i, a_)}, ensure_ascii=False) + "\n"); have[cname].add(i)
        fo.flush(); nb += 1
        if nb % 50 == 0: print(f"  {cname}: {len(have[cname])}/{len(R)} {time.time() - t0:.0f}s", flush=True)
    print(f"  {cname}: 끝 {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
