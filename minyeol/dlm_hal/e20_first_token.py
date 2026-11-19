"""E20 (PREREG.md): 첫 토큰 고정 진단.  행 S.  공여 토큰 = 같은 사실의 다른 문구 중 E7 첫 생성이 정답/오답인 행의 답 칸 0 확정 토큰.
usage: python followup_mk/e20_first_token.py [--limit N] [--verify]   --verify: e12 와 같은 묶음 구성으로 처음 4 묶음을 prefill 없이 돌려 results_e12_B 의 base 와 비교."""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import dloop_p2 as dl
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r"))
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]; tok0 = {int(r): int(committed[j, 0]) for j, r in enumerate(R7)}
m = Dream(); SP = m.SP; relx = {i: rows[i]["relation"] for i in range(len(rows))}
if "--verify" in sys.argv:
    _, _, F7 = e7_rows(); R = [int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]; ref = {}
    for l in open(f"{OUT}/results_e12_B.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["cond"] == "base": ref[d["row"]] = d["answer"]
    ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R}; same = tot = 0
    for c, (L, ix) in enumerate(by_length(R, lambda i: ids[i])):
        if c >= 4: break
        ans, _ = dl.generate(m, [ids[i] for i in ix]); tot += len(ix); same += sum(a == ref[i] for i, a in zip(ix, ans))
    print(f"검증: dloop_p2 (prefill 없음) 대 E12 base 일치 {same}/{tot}", flush=True); sys.exit(0)
S = set_S()
if LIM: S = [x for j, x in enumerate(S) if j % max(1, len(S) // LIM) == 0][:LIM]
st = {i: s for s, i in S}
ok_tok = lambda t: t not in SP and t != m.mask_id
byfact = collections.defaultdict(list); byrel = collections.defaultdict(list)
for r in R7:
    r = int(r)
    if r in meta and ok_tok(tok0[r]): byfact[fid[r]].append(r); byrel[relx[r]].append(r)
isc = lambda r: meta[r]["new_type"] == "correct"
def pick(pool, seed):
    return int(pool[np.random.default_rng(seed).integers(len(pool))]) if pool else None
donor = {}   # (cond, row) -> donor row
for s, i in S:
    f, rl = fid[i], relx[i]
    if s in ("broken", "entity"):
        cd = [j for j in byfact[f] if j != i and isc(j)]; dj = pick(cd, 11 * i + 1)
        if dj is not None:
            donor[("정답토큰고정", i)] = dj
            pool = [j for j in byrel[rl] if fid[j] != f and isc(j) and tok0[j] != tok0[dj]]; rj = pick(pool, 13 * i + 2)
            if rj is not None: donor[("무작위토큰고정", i)] = rj
    else:   # correct, five (zero 는 이번 진단에서 쓰지 않음)
        if s == "zero": continue
        wd = [j for j in byfact[f] if j != i and not isc(j) and meta[j]["new_type"] in ("entity", "broken")]
        if not wd: wd = [j for j in byrel[rl] if fid[j] != f and not isc(j) and meta[j]["new_type"] in ("entity", "broken")]
        dj = pick(wd, 17 * i + 3)
        if dj is not None: donor[("오답토큰고정", i)] = dj
conds = ["정답토큰고정", "무작위토큰고정", "오답토큰고정"]
print("공여 수", {c: sum(k_[0] == c for k_ in donor) for c in conds}, " 행 수", collections.Counter(s for s, _ in S), flush=True)
ids = {i: m.ids(build_prompt(rows[i]["question"])) for _, i in S}
fn = f"{OUT}/results_e20{'_pilot' if LIM else ''}.jsonl"; have = set()
if os.path.exists(fn):
    for l in open(fn, encoding="utf-8"):
        try: d = json.loads(l); have.add((d["cond"], d["row"]))
        except Exception: pass
fo = open(fn, "a", encoding="utf-8")
for cname in conds:
    rs = [i for _, i in S if (cname, i) in donor]; t0 = time.time()
    for L, ix in by_length(rs, lambda i: ids[i]):
        if all((cname, i) in have for i in ix): continue
        ba, _ = dl.generate(m, [ids[i] for i in ix])
        fa, _ = dl.generate(m, [ids[i] for i in ix], prefill=[tok0[donor[(cname, i)]] for i in ix])
        for i, a, b in zip(ix, ba, fa):
            dj = donor[(cname, i)]; base_rec = dict(cond=cname + "|같은묶음base", set=st[i], row=i, k=int(k[i]), answer=a, new_type=m.grade(i, a))
            rec = dict(cond=cname, set=st[i], row=i, k=int(k[i]), answer=b, new_type=m.grade(i, b), donor=dj, donor_tok=m.tok.decode([tok0[dj]]), donor_type=meta[dj]["new_type"], donor_same_fact=bool(fid[dj] == fid[i]))
            fo.write(json.dumps(base_rec, ensure_ascii=False) + "\n" + json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
    print(f"  {cname}: {len(rs)} 행 {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
