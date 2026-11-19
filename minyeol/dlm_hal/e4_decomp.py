"""E4 (PREREG.md): 두 단계 분해.  붕괴 행에 같은 사실 정답 문구의 상태를 주어 구간 (초기 층) 또는 프롬프트 끝에 넣는다.  usage: python followup_mk/e4_decomp.py [--limit N]"""
import collections, json, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
from transformers import AutoTokenizer
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
X, norm = load_X(); dirs = directions(X)
tq = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
def span(i):
    p = build_prompt(rows[i]["question"]); enc = tq(p, return_offsets_mapping=True); off = enc["offset_mapping"]; low = p.lower(); x = POP[rows[i]["sample_id"]]
    for nm in sorted([m_ for m_ in [x["subj"]] + json.loads(x["s_aliases"] or "[]") if m_], key=len, reverse=True):
        j = low.find(nm.lower())
        if j >= 0: return enc["input_ids"], [t for t, (a, b) in enumerate(off) if a < j + len(nm) and b > j]
    return enc["input_ids"], None
byf = collections.defaultdict(list)
for i in np.where(mid)[0]: byf[fid[i]].append(int(i))
SP_ = {}; T = []
for i in np.where(mid & (ty == "broken"))[0]:
    i = int(i); SP_[i] = span(i)
    if SP_[i][1] is None: continue
    cd = [j for j in byf[fid[i]] if ty[j] == "correct" and span(j)[1] is not None]; wd_ = [j for j in byf[fid[i]] if j != i and ty[j] != "correct" and span(j)[1] is not None]
    if cd: T.append((i, cd[0], wd_[0] if wd_ else None))
if LIM: T = T[:LIM]
donors = sorted({d for _, c, w in T for d in (c, w) if d is not None})
for d in donors: SP_[d] = span(d)
print(f"대상 {len(T)}, 오답 공여 있음 {sum(w is not None for _, _, w in T)}, 공여 {len(donors)}", flush=True)
m = Dream()
for i in list(SP_)[:200]: assert m.ids(build_prompt(rows[i]["question"])) == SP_[i][0], "토큰화 불일치"
# 공여 상태 캡처: step 0 입력 한 번 (steps=1)
NEED = list(range(0, 25)); cap = {}; pix = {}
for i in donors: kp = sorted(set(SP_[i][1]) | {len(SP_[i][0]) - 1}); pix[i] = {p: n_ for n_, p in enumerate(kp)}
for L, ix in by_length(donors, lambda i: SP_[i][0]):
    buf = {}
    def mk(l_):
        def f(h, step):
            if step == 0: buf[l_] = h[:, :L].detach().cpu()
        return f
    m.generate([SP_[i][0] for i in ix], {l_: mk(l_) for l_ in NEED}, steps=1)
    for j, i in enumerate(ix): cap[i] = {l_: buf[l_][j][list(pix[i])].clone() for l_ in NEED}
    del buf
def cv(d, l_, ps): return cap[d][l_][[pix[d][p] for p in ps]]
cs = []
for d in donors:
    v = cv(d, LAYER, [len(SP_[d][0]) - 1])[0].float(); cs.append(float(v @ torch.from_numpy(X[d]) / (v.norm() * np.linalg.norm(X[d]))))
print(f"캡처 층20 프롬프트 끝 대 저장 상태 cosine: 최소 {min(cs):.4f}, 평균 {np.mean(cs):.4f}", flush=True)
bad = {d for d, c_ in zip(donors, cs) if c_ < 0.99}; print(f"cosine < 0.99 공여 {len(bad)}개: {sorted(round(c_, 4) for c_ in cs if c_ < 0.99)}", flush=True)
T = [(i, c, (w if w not in bad else None)) for i, c, w in T if c not in bad]; print(f"정답 공여 점검 후 대상 {len(T)}, 오답 공여 있음 {sum(w is not None for _, _, w in T)}", flush=True)
nmis = sum(len(SP_[i][1]) != len(SP_[c][1]) for i, c, _ in T); print(f"주어 구간 길이가 다른 쌍 {nmis}/{len(T)}", flush=True)
def subj_edit(i, d, layers):
    n = min(len(SP_[i][1]), len(SP_[d][1])); tp = SP_[i][1][-n:]; dp = SP_[d][1][-n:]
    return {l_: (tp, cv(d, l_, dp)) for l_ in layers}
def end_edit(i, d, layers):
    return {l_: ([len(SP_[i][0]) - 1], cv(d, l_, [len(SP_[d][0]) - 1])) for l_ in layers}
R = lambda a, b: list(range(a, b + 1))
CONDS = [("base", None), ("END L20", lambda i, c, w: end_edit(i, c, [20])), ("SUBJ L0-8", lambda i, c, w: subj_edit(i, c, R(0, 8))),
         ("SUBJw L0-8", lambda i, c, w: subj_edit(i, w, R(0, 8)) if w is not None else {}),
         ("SUBJ+END", lambda i, c, w: {**subj_edit(i, c, R(0, 8)), **{20: end_edit(i, c, [20])[20]}}),
         ("B c1", "B"), ("D0 c1", "D0")] + \
        [(f"SUBJ L{a}-{b}", (lambda a, b: lambda i, c, w: subj_edit(i, c, R(a, b)))(a, b)) for a, b in ((0, 3), (4, 8), (9, 14), (15, 20))] + \
        [(f"END L{l_}", (lambda l_: lambda i, c, w: end_edit(i, c, [l_]))(l_)) for l_ in (4, 8, 12, 16, 24)]
meta = {i: (c, w) for i, c, w in T}; fo = open(f"{OUT}/results_e4{'_pilot' if LIM else ''}.jsonl", "w", encoding="utf-8")
for cname, spec in CONDS:
    t0 = time.time()
    for L, ix in by_length([i for i, _, _ in T], lambda i: SP_[i][0]):
        fns = {}
        if isinstance(spec, str):
            fns[LAYER] = add_end(torch.from_numpy(np.stack([dirs[grp[i]][spec] for i in ix])).float().cuda() * norm, L)
        elif spec is not None:
            per = collections.defaultdict(lambda: ([], [], []))
            for j, i in enumerate(ix):
                for l_, (pos, val) in spec(i, *meta[i]).items():
                    b_, p_, v_ = per[l_]; b_ += [j] * len(pos); p_ += list(pos); v_.append(val)
            for l_, (b_, p_, v_) in per.items():
                bt = torch.tensor(b_, device="cuda"); pt = torch.tensor(p_, device="cuda"); vt = torch.cat(v_).cuda()
                def f(h, step, bt=bt, pt=pt, vt=vt): h[bt, pt] = vt.to(h.dtype)
                fns[l_] = f
        ans = m.generate([SP_[i][0] for i in ix], fns)
        for i, a in zip(ix, ans): fo.write(json.dumps({"cond": cname, "row": i, "k": int(k[i]), "donor": meta[i][0], "wdonor": meta[i][1], "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n")
    fo.flush(); print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
