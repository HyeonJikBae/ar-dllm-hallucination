"""E52 (PREREG.md '## E52 내부 처리: 주의와 헤드').  행 = 1~4/5 5,950 (np.where(mid) 순서, by_length 묶음 16).  조건마다 같은 묶음 구성.
--mode gen : 끝까지 생성.  조건 base (주의 기록), ko_{all,0_9,10_19,20_27} (프롬프트 끝 -> 확정 칸 주의 막기, step>=2, 칸 0 확정 전까지),
             rnd_{all,...} (같은 수의 마스크 칸 막기, 대조), empty / empty_eager (sanity), brel / rand (층 20 프롬프트 끝 조향 c1, 모든 step, 주의 기록).
--mode dla : step 0 forward 한 번 (조건 base / brel / rand).  헤드별 DLA (금, 경쟁), 주어 끝·캔버스 주의.  경쟁 토큰은 followup_mk/e52/comp.json.
usage: python followup_mk/e52_run.py --mode gen --conds base,ko_all [--limit NB] [--shard s --nshards n]
재개 가능: 결과 jsonl 에 이미 있는 행의 묶음은 건너뜀."""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
import e52_lib as E

arg = lambda k_, d=None: sys.argv[sys.argv.index(k_) + 1] if k_ in sys.argv else d
MODE = arg("--mode", "gen"); CN = arg("--conds", "base").split(","); LIM = int(arg("--limit", 0)); SH = int(arg("--shard", 0)); NS = int(arg("--nshards", 1))
SUB = arg("--sub")   # 부분 집합 실행: followup_mk/e52/{SUB}/rows.json 의 행만 (자체 묶음 구성; 기준 조건도 같은 구성으로 같이 돌림)
DIR = f"{OUT}/e52{'_pilot' if LIM else ''}" + (f"/{SUB}" if SUB else ""); os.makedirs(DIR, exist_ok=True)
POS = {int(k_): v for k_, v in json.load(open(f"{OUT}/e52/pos.json")).items()}
R = [int(i) for i in np.where(mid)[0]] if not SUB else [int(i) for i in json.load(open(f"{DIR}/rows.json"))]

m = Dream(); E.patch(m); tok = m.tok
if float(arg("--reserve_gb", 0)) > 0: _res = torch.empty(int(float(arg("--reserve_gb", 0)) * 2**30), dtype=torch.uint8, device="cuda"); del _res   # 공유 GPU: 필요한 만큼 캐시에 미리 잡아 둠 (다른 작업이 채워 OOM 나는 것 방지)
ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in R}
bad = [i for i in R if ids[i] != POS[i]["ids"]]
print(f"Dream 토큰 = Qwen offset 토큰: {len(R) - len(bad)}/{len(R)} 일치", flush=True)
gold = {i: tok(" " + rows[i]["gold_value"], add_special_tokens=False)["input_ids"][0] for i in R}
batches = list(by_length(R, lambda i: ids[i]))
if LIM: batches = batches[::max(1, len(batches) // LIM)][:LIM]
mine = [b for j, b in enumerate(batches) if j % NS == SH]
rows_mine = [i for _, ix in mine for i in ix]; n = len(rows_mine); where = {i: j for j, i in enumerate(rows_mine)}

STEER = {}
if any(c in ("brel", "rand") for c in CN):
    X, norm = load_X(); dirs = directions(X); rel = np.array([r["relation"] for r in rows])
    BREL = {}
    for g in range(NF):   # e12_regen.py 와 같은 관계별 방향 (주어 겹밖)
        tr = grp != g; byf = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in np.where(tr & mid & (ty != "abstain"))[0]: byf[fid[i]]["c" if ty[i] == "correct" else "w"].append(i)
        per = collections.defaultdict(list)
        for f_, v in byf.items():
            if v["c"] and v["w"]: per[rows[v["c"][0]]["relation"]].append(X[v["c"]].mean(0) - X[v["w"]].mean(0))
        BREL[g] = {r_: unit(np.mean(d, 0)) for r_, d in per.items() if len(d) >= 15}
    del X
    STEER["brel"] = lambda ix: np.stack([BREL[grp[i]].get(rel[i], dirs[grp[i]]["B"]) for i in ix]) * norm
    STEER["rand"] = lambda ix: np.stack([unit(np.random.default_rng(10_000 + int(i)).standard_normal(3584)) for i in ix]) * norm
    print(f"조향 크기 norm {norm:.2f}", flush=True)

def steer_fns(c, ix, L):
    if c not in STEER: return {}
    V = torch.from_numpy(STEER[c](ix)).float().cuda()
    def f(h, step, V=V, L=L): h[:, L - 1] = (h[:, L - 1].float() + V).to(h.dtype)
    return {LAYER: f}

def spec(c):
    if c in ("base", "basesub"): return dict(cap=True)
    if c in ("brel", "rand"): return dict(cap=True)
    if c in ("empty", "empty_eager"): return dict(knock=c, layers="all")
    a, b = c.split("_", 1)
    return dict(knock={"ko": "committed", "rnd": "random"}[a], layers=b)

mm = lambda name, shape, dt: np.lib.format.open_memmap(name, mode="r+" if os.path.exists(name) else "w+", dtype=dt, shape=shape)
COMP = json.load(open(f"{DIR}/comp.json")) if MODE == "dla" else None
print(f"mode {MODE}, 조건 {CN}, shard {SH}/{NS}: 묶음 {len(mine)}, 행 {n}", flush=True)
for c in CN:
    fout = f"{DIR}/{MODE}_{c}_s{SH}.jsonl"; have = set()
    if os.path.exists(fout):
        for l in open(fout, encoding="utf-8"):
            try: have.add(json.loads(l)["row"])
            except Exception: pass
    fo = open(fout, "a", encoding="utf-8"); t0 = time.time(); nb = 0
    if MODE == "gen":
        sp = spec(c); ATT = mm(f"{DIR}/att_{c}_s{SH}.npy", (n, 3, 28, 28), np.float16) if sp.get("cap") else None
    else:
        DLA = mm(f"{DIR}/dla_{c}_s{SH}.npy", (n, 6, 28, 28), np.float32)   # dla_g, dla_c, att_subj, att_canvas, att_qlast, att_pos0
    for L, ix in mine:
        if all(i in have for i in ix): continue
        fns = steer_fns(c, ix, L); g_ = [gold[i] for i in ix]
        if MODE == "gen":
            r = E.generate(m, [ids[i] for i in ix], g_, fns=fns, seed=ix[0], **sp)
            if ATT is not None:
                A = r["att_comm"]
                for j, i in enumerate(ix):
                    la = r["s0_last"][j]; a2 = A[j, 2]; al = A[j, la] if la >= 2 else np.zeros((28, 28), np.float32)
                    am = A[j, 2:la + 1].mean(0) if la >= 2 else np.zeros((28, 28), np.float32)
                    ATT[where[i]] = np.stack([a2, al, am]).astype(np.float16)
                ATT.flush()
            for j, i in enumerate(ix):
                fo.write(json.dumps({"cond": c, "row": i, "answer": r["answers"][j], "new_type": m.grade(i, r["answers"][j]), "gold_id": int(gold[i]),
                                     "s0_last": r["s0_last"][j], "s0_tok": r["s0_tok"][j], "rank": r["rank"][j].tolist(), "nk": r["nk"][j].tolist(),
                                     "canvas": r["canvas"][j].tolist()}, ensure_ascii=False) + "\n")
        else:
            cp = [int(COMP.get(str(i), -1)) for i in ix]; sj = [POS[i]["se"] if POS[i]["se"] is not None else -1 for i in ix]
            r = E.step0(m, [ids[i] for i in ix], g_, cp, sj, fns=fns, qlast=[POS[i]["ql"] for i in ix])
            for j, i in enumerate(ix):
                DLA[where[i]] = np.stack([r["dla_g"][j], r["dla_c"][j], r["att_subj"][j], r["att_canvas"][j], r["att_ql"][j], r["att_p0"][j]])
            DLA.flush()
            for j, i in enumerate(ix):
                fo.write(json.dumps({"cond": c, "row": i, "comp": cp[j], "subj": sj[j], "L": L, "scale": float(r["scale"][j]), "logit_diff": float(r["logit_diff"][j]),
                                     "lin_diff": float(r["lin_diff"][j]), "rank0": int(r["rank0"][j]), "top1": int(r["top1"][j])}) + "\n")
        fo.flush(); have.update(ix); nb += 1
        if nb % 25 == 0: print(f"  {c}: {len(have)}/{n} {time.time() - t0:.0f}s", flush=True)
    json.dump({"rows": rows_mine}, open(f"{DIR}/rows_{MODE}_s{SH}.json", "w"))
    print(f"  {c}: 끝 {len(have)}/{n} {time.time() - t0:.0f}s, 최대 할당 {torch.cuda.max_memory_allocated() / 2**30:.1f}GB 예약 {torch.cuda.memory_reserved() / 2**30:.1f}GB", flush=True)
print("DONE", flush=True)
