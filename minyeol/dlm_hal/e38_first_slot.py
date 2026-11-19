"""E38 (PREREG.md): 첫 칸 겨냥 조향과 첫 칸 확정 시점 조절.  행 S (1,510), e9 와 같은 묶음 구성.
usage: python followup_mk/e38_first_slot.py [--limit N] [--conds A,B,...]
조건: A base | B 프롬프트 끝 (L-1) Brel c1 s0-3 | C05/C1 답 칸 0 위치 (L) Brel c0.5/c1 s0-3 | D 답 칸 0 위치 무작위 c1 s0-3 |
      E1/E2 칸 0 로짓 + lambda z(W_U Brel) s0-3 | F 칸 0 을 step 4 에 확정 | G F + B
주의 (구현 확인): Dream 은 로짓을 한 칸 이동하므로 칸 0 (위치 L) 의 로짓은 위치 L-1 (프롬프트 끝) 의 출력이다.  즉 B 는 칸 0 을 읽는 위치를 조향하고, C/D (위치 L) 는 칸 1 을 읽는 위치를 조향한다."""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import dloop_p8 as dl
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
ALL = ["A", "B", "C05", "C1", "D", "E1", "E2", "F", "G"]
CONDS = sys.argv[sys.argv.index("--conds") + 1].split(",") if "--conds" in sys.argv else ALL
assert all(c in ALL for c in CONDS), CONDS
X, norm = load_X(); dirs = directions(X); rel = np.array([r["relation"] for r in rows])
BREL = {}
for g in range(NF):   # e12_regen.py 와 같은 관계별 방향
    tr = grp != g; byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.where(tr & mid & (ty != "abstain"))[0]: byf[fid[i]]["c" if ty[i] == "correct" else "w"].append(i)
    per = collections.defaultdict(list)
    for f_, v in byf.items():
        if v["c"] and v["w"]: per[rows[v["c"][0]]["relation"]].append(X[v["c"]].mean(0) - X[v["w"]].mean(0))
    BREL[g] = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}
brel = lambda i: BREL[grp[i]].get(rel[i], dirs[grp[i]]["B"])
rand = lambda i: unit(np.random.default_rng(20_000 + int(i)).standard_normal(X.shape[1]))
S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}; R = [i for _, i in S]
m = Dream(); ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R}
gold = {i: m.tok(" " + rows[i]["gold_value"], add_special_tokens=False)["input_ids"][0] for i in R}
WU = m.model.lm_head.weight   # [V, D] (bf16)
_zc = {}
def zbias(i):
    v = brel(i); key = v.tobytes()
    if key not in _zc:
        b = (WU.float() @ torch.from_numpy(v).float().cuda()); _zc[key] = ((b - b.mean()) / b.std())
    return _zc[key]
cu = lambda a: torch.from_numpy(np.stack(a)).float().cuda()
def kwargs(c, ix):
    if c == "A": return {}
    if c == "B": return dict(steer=cu([brel(i) for i in ix]) * norm, steer_pos="end")
    if c in ("C05", "C1"): return dict(steer=cu([brel(i) for i in ix]) * norm * (0.5 if c == "C05" else 1.0), steer_pos="slot0")
    if c == "D": return dict(steer=cu([rand(i) for i in ix]) * norm, steer_pos="slot0")
    if c in ("E1", "E2"): return dict(slot0_logit_bias=torch.stack([zbias(i) for i in ix]) * (1.0 if c == "E1" else 2.0))
    if c == "F": return dict(commit_at=4)
    if c == "G": return dict(commit_at=4, steer=cu([brel(i) for i in ix]) * norm, steer_pos="end")
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
fout = f"{OUT}/results_e38{'_pilot' if LIM else ''}{'_' + TAG if TAG else ''}.jsonl"; have = collections.defaultdict(set)
if os.path.exists(fout):
    for l in open(fout, encoding="utf-8"):
        try: d = json.loads(l); have[d["cond"]].add(d["row"])
        except Exception: pass
fo = open(fout, "a", encoding="utf-8")
print(f"행 {len(R)}, 조건 {CONDS}, norm {norm:.1f}, 관계별 방향 수 {[len(BREL[g]) for g in range(NF)]}", flush=True)
for c in CONDS:
    t0 = time.time(); nb = 0
    for L, ix in by_length(R, lambda i: ids[i]):
        if all(i in have[c] for i in ix): continue
        ans, stt = dl.generate(m, [ids[i] for i in ix], gold=[gold[i] for i in ix], **kwargs(c, ix))
        for j, (i, a) in enumerate(zip(ix, ans)):
            if i in have[c]: continue
            r = stt["gold_rank"][j]
            fo.write(json.dumps({"cond": c, "set": st[i], "row": int(i), "k": int(k[i]), "answer": a, "new_type": m.grade(i, a), **{f"gold_rank_s{s_}": r[s_] for s_ in range(4)}}, ensure_ascii=False) + "\n"); have[c].add(i)
        fo.flush(); nb += 1
    print(f"  {c}: {time.time() - t0:.0f}s ({nb} 묶음)", flush=True)
print("DONE", flush=True)
