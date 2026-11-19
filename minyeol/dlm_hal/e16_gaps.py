"""E16 (PREREG.md): GAPS (arXiv 2609.01878) 식 차원 게이트를 층 20 프롬프트 끝 조향에 붙임.  행 S (1,510).
usage: python followup_mk/e16_gaps.py --conds "base,B c1,..." --tag A [--limit N]"""
import collections, json, os, sys, time
import numpy as np, torch
from scipy.stats import rankdata
sys.path.insert(0, "followup_mk")
from common import *
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "A"
CONDS_ALL = {"base": None, "B c0.5": ("dense", "B", .5, None), "B c1": ("dense", "B", 1., None), "Brel c1": ("dense", "Brel", 1., None),
             "정적 B c1": ("gaps", "B", 1., "static"), "정적 B c2": ("gaps", "B", 2., "static"),
             "정적+동적 B c1": ("gaps", "B", 1., "sp"), "정적+동적 B c2": ("gaps", "B", 2., "sp"), "정적+동적 B c4": ("gaps", "B", 4., "sp"),
             "정적+동적 Brel c1": ("gaps", "Brel", 1., "sp"), "정적+동적 Brel c2": ("gaps", "Brel", 2., "sp"),
             "변형 Brel c1": ("gaps", "Brel", 1., "ours"), "변형 Brel c2": ("gaps", "Brel", 2., "ours"), "무작위 B c2": ("gaps", "B", 2., "rand")}
CONDS = [c for c in sys.argv[sys.argv.index("--conds") + 1].split(",")] if "--conds" in sys.argv else list(CONDS_ALL)
assert all(c in CONDS_ALL for c in CONDS), CONDS
X, norm = load_X(); dirs = directions(X); rel = np.array([r["relation"] for r in rows]); D = X.shape[1]
lab_ok = mid & (ty != "abstain"); yc = ty == "correct"
# ---- 방향 (e9 와 같음)
BREL = {}
for g in range(NF):
    tr = grp != g; byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & lab_ok)[0]: byf[fid[i]]["c" if yc[i] else "w"].append(i)
    per = collections.defaultdict(list)
    for f_, v in byf.items():
        if v["c"] and v["w"]: per[rows[v["c"][0]]["relation"]].append(X[v["c"]].mean(0) - X[v["w"]].mean(0))
    BREL[g] = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}
# ---- 게이트 통계 (겹마다)
ST = {}
for g in range(NF):
    t0 = time.time(); tr = (grp != g) & lab_ok; Xt = X[tr]; y = yc[tr]; n1, n2 = int((~y).sum()), int(y.sum())
    R_ = rankdata(Xt, axis=0); auc = (R_[~y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n2)          # P(오답 > 정답) (src = 오답)
    s0 = np.sqrt((n1 + n2 + 1) / (12 * n1 * n2)); msep = np.abs(auc - .5) > 7 * s0
    mw, sw, mc, sc_ = Xt[~y].mean(0), Xt[~y].std(0) + 1e-4, Xt[y].mean(0), Xt[y].std(0) + 1e-4
    byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr)[0]: byf[fid[i]]["c" if yc[i] else "w"].append(i)
    Df = np.stack([X[v["c"]].mean(0) - X[v["w"]].mean(0) for v in byf.values() if v["c"] and v["w"]]); tt = Df.mean(0) / (Df.std(0) / np.sqrt(len(Df)) + 1e-9)
    relst = {}
    for r_ in np.unique(rel[tr]):
        a = tr & (rel == r_) & ~yc; b = tr & (rel == r_) & yc
        if a.sum() >= 30 and b.sum() >= 30: relst[r_] = (X[a].mean(0), X[a].std(0) + 1e-4, X[b].mean(0), X[b].std(0) + 1e-4)
    ST[g] = dict(msep=msep, glob=(mw, sw, mc, sc_), tmask=np.abs(tt) > 5, rel=relst)
    print(f"fold {g}: 정적 GAPS 켜짐 {int(msep.sum())}/{D} (기준 |AUROC-0.5|>{7 * s0:.4f}), |t|>5 {int((np.abs(tt) > 5).sum())}, 관계 통계 {len(relst)}개, {time.time() - t0:.0f}s", flush=True)
FULL = "--rows" in sys.argv and sys.argv[sys.argv.index("--rows") + 1] == "full"
if FULL:   # E25: E12 와 같은 행 (1~4/5 5,950 + 5/5 2,000)
    _, _, F7_ = e7_rows(); S = [("mid", int(i)) for i in np.where(mid)[0]] + [("five", int(i)) for i in F7_]
else: S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}
out_f = f"{OUT}/results_e16{'_full' if FULL else ''}{'_pilot' if LIM else ''}_{TAG}.jsonl"; have = collections.defaultdict(set)
if os.path.exists(out_f):
    for l in open(out_f, encoding="utf-8"):
        try: d = json.loads(l); have[d["cond"]].add(d["row"])
        except Exception: pass
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}
fo = open(out_f, "a", encoding="utf-8"); cu = lambda a: torch.from_numpy(np.stack(a)).float().cuda()
def batch_tensors(ix, kind, gate):
    v = [BREL[grp[i]].get(rel[i], dirs[grp[i]]["B"]) if kind == "Brel" else dirs[grp[i]]["B"] for i in ix]; P = {"v": cu(v)}
    if gate in ("static", "sp", "rand"): P["msep"] = cu([ST[grp[i]]["msep"].astype(np.float32) for i in ix])
    if gate == "ours": P["msep"] = cu([ST[grp[i]]["tmask"].astype(np.float32) for i in ix])
    if gate in ("sp", "rand"): gs = [ST[grp[i]]["glob"] for i in ix]
    elif gate == "ours": gs = [ST[grp[i]]["rel"].get(rel[i], ST[grp[i]]["glob"]) for i in ix]
    else: gs = None
    if gs is not None: P["mw"], P["sw"], P["mc"], P["sc"] = (cu([x[q] for x in gs]) for q in range(4))
    return P
for cname in CONDS:
    spec = CONDS_ALL[cname]; t0 = time.time(); nb = 0
    for L, ix in by_length([i for _, i in S], lambda i: ids[i]):
        if all(i in have[cname] for i in ix): continue
        fns = {}; non = {}
        if spec is not None:
            op, kind, c, gate = spec; P = batch_tensors(ix, kind, gate); seeds = torch.tensor([int(i) for i in ix])
            def f(h, step, P=P, L=L, c=c, gate=gate, op=op, ix=ix, seeds=seeds, non=non):
                x = h[:, L - 1].float()
                if op == "dense": delta = c * norm * P["v"]
                else:
                    msk = P["msep"].clone()
                    if gate in ("sp", "rand", "ours"):
                        lw = -torch.log(P["sw"]) - (x - P["mw"]) ** 2 / (2 * P["sw"] ** 2); lc = -torch.log(P["sc"]) - (x - P["mc"]) ** 2 / (2 * P["sc"] ** 2)
                        msk = msk * (lw > lc).float()
                    if gate == "rand":
                        cnt = msk.sum(1).long().tolist(); msk = torch.zeros_like(msk)
                        for j, n_ in enumerate(cnt):
                            gen = torch.Generator(device=x.device); gen.manual_seed(int(seeds[j]) * 100 + step)
                            msk[j, torch.randperm(x.shape[1], generator=gen, device=x.device)[:n_]] = 1.
                    if step == 0:
                        for j, i in enumerate(ix): non[i] = int(msk[j].sum().item())
                    delta = c * norm * msk * P["v"]
                h[:, L - 1] = (x + delta).to(h.dtype)
            fns[LAYER] = f
        ans = m.generate([ids[i] for i in ix], fns)
        for i, a_ in zip(ix, ans):
            fo.write(json.dumps({"cond": cname, "set": st[i], "row": int(i), "k": int(k[i]), "answer": a_, "new_type": m.grade(i, a_), "n_on": non.get(i)}, ensure_ascii=False) + "\n")
        fo.flush(); nb += 1
    print(f"  {cname}: {time.time() - t0:.0f}s ({nb} 묶음)", flush=True)
print("DONE", flush=True)
