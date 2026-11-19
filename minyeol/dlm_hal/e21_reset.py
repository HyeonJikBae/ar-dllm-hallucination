"""E21 (PREREG.md): 되돌리기 범위.  틀린 첫 생성 (E7 최종 캔버스) 에서 (a) 답 칸 0 (b) 답 내용 칸 (c) 캔버스 전체 를 다시 가려 조향 없음 / Brel c1 로 다시 디코딩.
usage: python followup_mk/e21_reset.py [--limit N]   결과: results_e21[_pilot].jsonl (재개 가능)"""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import dloop_p3 as dl
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
X, norm = load_X(); dirs = directions(X); rel = np.array([r["relation"] for r in rows])
BREL = {}
for g in range(NF):   # e12_regen.py 와 같은 관계별 방향
    tr = grp != g; byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & (ty != "abstain"))[0]: byf[fid[i]]["c" if ty[i] == "correct" else "w"].append(i)
    per = collections.defaultdict(list)
    for f_, v in byf.items():
        if v["c"] and v["w"]: per[rows[v["c"][0]]["relation"]].append(X[v["c"]].mean(0) - X[v["w"]].mean(0))
    BREL[g] = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}
def vecs(ix): return np.stack([BREL[grp[i]].get(rel[i], dirs[grp[i]]["B"]) for i in ix])
# ---- E7 첫 생성 캔버스 복원 (e10_detect.py 와 같은 방식)
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
p7 = {int(r): j for j, r in enumerate(R7)}
S = [(s, i) for s, i in set_S() if s in ("broken", "entity")]; n_all = len(S)
S = [(s, i) for s, i in S if i in p7 and meta[i]["new_type"] != "correct"]
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
print(f"S 붕괴+entity {n_all}, E7 에 있고 첫 생성이 오답 {len(S)}", flush=True)
J = np.array([p7[i] for _, i in S]); IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J])
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); CAN = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]   # [n, 32] 확정 토큰
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}; pos = {i: j for j, (_, i) in enumerate(S)}; st = {i: s for s, i in S}
SPt = np.array(sorted(m.SP))
def content_mask(c):   # 첫 특수 토큰 앞 칸 (없으면 전부, 첫 칸이 특수면 칸 0 만)
    isp = np.isin(c, SPt); f = int(isp.argmax()) if isp.any() else len(c); f = max(f, 1); mk = np.zeros(len(c), bool); mk[:f] = True; return mk
# 복원 점검: 확정 토큰으로 디코딩한 답 = E7 답
def dec(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in m.SP), len(t))]; return extract_answer(m.tok.decode(t, skip_special_tokens=True))
ok = np.mean([dec(CAN[pos[i]]) == meta[i]["answer"] for _, i in S]); print(f"캔버스 복원 답 = E7 답: {100 * ok:.2f}%", flush=True)
assert ok > 0.99, "복원 불일치"
COND = [("a 칸0 | 없음", "a", 0.), ("a 칸0 | Brel c1", "a", 1.), ("b 답구간 | 없음", "b", 0.), ("b 답구간 | Brel c1", "b", 1.), ("c 전체 | 없음", "c", 0.), ("c 전체 | Brel c1", "c", 1.)]
fout = f"{OUT}/results_e21{'_pilot' if LIM else ''}.jsonl"; have = collections.defaultdict(set)
if os.path.exists(fout):
    for l in open(fout, encoding="utf-8"):
        try: d = json.loads(l); have[d["cond"]].add(d["row"])
        except Exception: pass
fo = open(fout, "a", encoding="utf-8")
for cname, scope, c in COND:
    t0 = time.time()
    for L, ix in by_length([i for _, i in S], lambda i: ids[i]):
        if all(i in have[cname] for i in ix): continue
        fns = {}
        if c > 0:
            V = torch.from_numpy(vecs(ix)).float().cuda()
            def f(h, step, V=V, L=L, c=c): h[:, L - 1] = (h[:, L - 1].float() + V * c * norm).to(h.dtype)
            fns[LAYER] = f
        if scope == "c":     # 처음부터 다시 생성 (32 step, 원본 규칙과 같은 고리)
            ans, _ = dl.generate(m, [ids[i] for i in ix], fns, mode="orig")
        else:
            can = torch.from_numpy(np.stack([CAN[pos[i]] for i in ix])); rs = np.zeros(can.shape, bool)
            for j, i in enumerate(ix):
                if scope == "a": rs[j, 0] = True
                else: rs[j] = content_mask(CAN[pos[i]])
            ans, _ = dl.resume(m, [ids[i] for i in ix], can, torch.from_numpy(rs), steps=8, fns=fns)
        for i, a in zip(ix, ans):
            if i in have[cname]: continue
            fo.write(json.dumps({"cond": cname, "set": st[i], "row": i, "k": int(k[i]), "first_type": meta[i]["new_type"], "first_answer": meta[i]["answer"], "answer": a, "new_type": m.grade(i, a)}, ensure_ascii=False) + "\n"); have[cname].add(i)
        fo.flush()
    print(f"  {cname}: {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
