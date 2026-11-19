"""E54 (PREREG.md) 생성 (GPU).  common.Dream.generate (원본 diffusion_generate, topk_margin, 32/32) + 층 20/24 출력의 프롬프트 끝 (L-1) 훅.
같은 수신 행의 모든 조건 (base 포함) 을 한 묶음에 넣는다 (bf16 묶음 효과 통제: 같은 묶음 base).
--part a : 확정 전 교환.  짝 = followup_mk/e54/pairs.json (E51 과 같은 규칙).  step 0~1 (캔버스 전부 마스크) 에만.
    정방향 (수신 F, 공여 S): base, full20/full24 (상태 전체를 S 의 E44 step 0 상태로), pc{r}_{L} (겹밖 같은 사실 차이 SVD 상위 r 부분공간만 교환),
    rnd{r}_{L} (같은 rank 무작위 부분공간), r = 1, 4, 16, 64.
    역방향 (수신 S, 공여 F): base, full20/full24, pc{r}_{L}, rnd{16,64}_{L}.
--part b : 지우기.  행 = 1~4/5 5,950 + E7 5/5 앞 1,000 + 0/5 앞 1,000.  조건: base, leace{L}_{s03|all}, inlp{L}_{s03|all}, rnd1_{L}_all, rndr_{L}_all.
    지우개는 e54_fit.py (행의 주어 겹 g 를 뺀 겹으로 맞춘 것).
기록: followup_mk/results_e54_{part}[_pilot]_s{shard}.jsonl (행·조건별 답, 채점, step 1/2 칸 0 금 순위·1 위, 최종 칸 0 토큰).  재개 가능.
usage: python followup_mk/e54_run.py --part a|b [--pilot N] [--shard s --nshards n] [--nseq 40]"""
import collections, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "followup_mk")
from common import *
arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
PART = arg("--part", "a"); PIL = int(arg("--pilot", "0")); SH = int(arg("--shard", "0")); NS = int(arg("--nshards", "1")); NSEQ = int(arg("--nseq", "40"))
SFX = "_pilot" if PIL else ""
FN = f"{OUT}/results_e54_{PART}{SFX}_s{SH}.jsonl"
LAYS = (20, 24); RANKS = (1, 4, 16, 64)

# ---------------- E44 step 0 상태 (공여)
def e44_states(need):
    X = {L_: {} for L_ in LAYS}
    for d in sorted(x for x in os.listdir(f"{OUT}/e44") if x.startswith("shard")):
        D = f"{OUT}/e44/{d}"; R = np.load(f"{D}/rows.npy"); lay = json.load(open(f"{D}/config.json"))["layers"]; H = np.load(f"{D}/hid.npy", mmap_mode="r")
        for j, r in enumerate(R):
            if int(r) in need:
                for L_ in LAYS: X[L_][int(r)] = np.asarray(H[j, 0, lay.index(L_)], dtype=np.float32)
    return X

FIT = {L_: dict(np.load(f"{OUT}/e54/fit_L{L_}.npz")) for L_ in LAYS}
items = []   # dict(key, row, cond, role, donor, layer, steps, kind, r)
if PART == "a":
    pairs = json.load(open(f"{OUT}/e54/pairs.json"))
    if PIL:
        fs = sorted({p["fact"] for p in pairs}); keep = set(np.random.default_rng(54).permutation(fs)[:PIL].tolist()); pairs = [p for p in pairs if p["fact"] in keep]
    seen = set()
    for p in pairs:
        for role, rc, dn in (("F", p["F"], p["S"]), ("S", p["S"], p["F"])):
            # 비용 때문에 역방향은 r 4, 64 와 무작위 64 만 (결과 보기 전 결정, 2026-11-19 06:00)
            conds = [("base", None, None, None)] + [(f"full{L_}", L_, "full", None) for L_ in LAYS] + [(f"pc{r}_{L_}", L_, "pc", r) for L_ in LAYS for r in (RANKS if role == "F" else (4, 64))]
            conds += [(f"rnd{r}_{L_}", L_, "rnd", r) for L_ in LAYS for r in (RANKS if role == "F" else (64,))]
            if PIL: conds += [(f"self{L_}", L_, "self", None) for L_ in LAYS]
            for c, L_, kind, r in conds:
                key = (role, rc, "-" if c == "base" else dn, c)
                if key in seen: continue
                seen.add(key); items.append(dict(key=key, row=rc, role=role, donor=None if c == "base" else dn, pid=p["pid"], route=p["route"], cond=c, layer=L_, kind=kind, r=r, steps=(0, 1)))
    need = {p[x] for p in pairs for x in ("F", "S")}
else:
    _, Z7, F7 = e7_rows()
    R = [int(i) for i in np.where(mid)[0]] + [int(i) for i in F7[:1000]] + [int(i) for i in Z7[:1000]]
    if PIL: R = [int(i) for i in np.random.default_rng(54).permutation(np.where(mid)[0])[:PIL]] + [int(i) for i in F7[:PIL // 4]] + [int(i) for i in Z7[:PIL // 4]]
    ALL = tuple(range(STEPS)); S03 = (0, 1, 2, 3)
    conds = [("base", None, None, None, ())]
    for L_ in LAYS:
        # 비용 때문에 (결과 보기 전 결정, 2026-11-19 06:00): INLP 는 모든 step 만, 무작위 대조는 1~4/5 행에만
        conds += [(f"leace{L_}_s03", L_, "leace", 1, S03), (f"leace{L_}_all", L_, "leace", 1, ALL), (f"inlp{L_}_all", L_, "inlp", None, ALL),
                  (f"rnd1_{L_}_all", L_, "rndp", 1, ALL), (f"rndr_{L_}_all", L_, "rndp", None, ALL)]
    for i in R:
        for c, L_, kind, r, st in conds:
            if kind == "rndp" and not mid[i]: continue
            items.append(dict(key=("B", i, "-", c), row=i, role="B", donor=None, cond=c, layer=L_, kind=kind, r=r, steps=st))
    need = set(R) if PIL else set()
X44 = e44_states(need) if need else None

def readj(fn):
    out = []
    if os.path.exists(fn):
        for l in open(fn, encoding="utf-8"):
            try: out.append(json.loads(l))
            except Exception: pass
    return out
import glob as _g   # 조각 수를 바꿔 다시 띄울 때 겹치지 않도록 모든 조각 파일을 읽어 이미 한 항목을 건너뜀
have = {(d["role"], d["row"], d["donor"] if d["donor"] is not None else "-", d["cond"]) for fn_ in _g.glob(f"{OUT}/results_e54_{PART}{SFX}_s*.jsonl") for d in readj(fn_)}

m = Dream(); tok = m.tok
ids = {i: m.ids(build_prompt(rows[i]["question"])) for i in {it["row"] for it in items}}
gold = {i: tok(" " + rows[i]["gold_value"], add_special_tokens=False)["input_ids"][0] for i in ids}
# 수신 행 단위로 묶고 (같은 행의 조건은 한 묶음), 같은 길이끼리 NSEQ 안에서 채움
byrow = collections.defaultdict(list)
for it in items: byrow[(it["role"], it["row"])].append(it)
byL = collections.defaultdict(list)
for kr, v in sorted(byrow.items()): byL[len(ids[kr[1]])].append(kr)
batches = []
for L, krs in sorted(byL.items()):
    cur = []
    for kr in krs:
        if cur and sum(len(byrow[x]) for x in cur) + len(byrow[kr]) > NSEQ: batches.append((L, cur)); cur = []
        cur.append(kr)
    if cur: batches.append((L, cur))
mine = [b for j, b in enumerate(batches) if j % NS == SH]
print(f"part {PART}: 항목 {len(items)}, 묶음 {len(batches)} (이 조각 {len(mine)}), 이미 {len(have)}", flush=True)
dev = "cuda"
G = {L_: {k_: torch.tensor(v, device=dev) for k_, v in FIT[L_].items()} for L_ in LAYS}

def make_hooks(seq, L):
    """seq: 이 묶음의 항목 목록 (순서 = 묶음 행).  층별 훅: 조건 무리별로 벡터화."""
    groups = collections.defaultdict(list)
    for j, it in enumerate(seq):
        if it["kind"] is None: continue
        g = int(grp[it["row"]]); groups[(it["layer"], it["kind"], it["r"], g, it["steps"])].append(j)
    don = {}
    for j, it in enumerate(seq):
        if it["kind"] in ("full", "pc", "rnd"): don[j] = X44[it["layer"]][it["donor"]]
        elif it["kind"] == "self": don[j] = X44[it["layer"]][it["row"]]
    GT = []
    for (L_, kind, r, g, st), js in groups.items():
        jt = torch.tensor(js, device=dev); P = G[L_]
        D_t = torch.tensor(np.stack([don[j] for j in js]), device=dev) if kind in ("full", "pc", "rnd", "self") else None
        if kind in ("full", "self"): op = lambda x, D_t=D_t: D_t
        elif kind in ("pc", "rnd"):
            U = P[f"U_{g}"][:r] if kind == "pc" else P["rand"][:r]
            op = lambda x, D_t=D_t, U=U: x + ((D_t - x) @ U.T) @ U
        elif kind == "leace":
            c, w, mm_ = P[f"leace_c_{g}"], P[f"leace_w_{g}"], P[f"leace_m_{g}"]; wc = (w @ c)
            op = lambda x, c=c, w=w, mm_=mm_, wc=wc: x - ((x @ w - mm_) / wc)[:, None] * c[None]
        elif kind in ("inlp", "rndp"):
            Q = P[f"inlp_Q_{g}"] if kind == "inlp" else P["rand"][: (r if r is not None else P[f"inlp_Q_{g}"].shape[0])]; mu = P[f"inlp_mu_{g}"]
            op = lambda x, Q=Q, mu=mu: x - ((x - mu) @ Q.T) @ Q
        GT.append((L_, set(st), jt, op))
    def mk(L_):
        def f(h, step):
            for (LL, st, jt, op) in GT:
                if LL == L_ and step in st:
                    x = h[jt, L - 1].float(); h[jt, L - 1] = op(x).to(h.dtype)
        return f
    return {L_: mk(L_) for L_ in LAYS if any(g[0] == L_ for g in GT)}

fo = open(FN, "a", encoding="utf-8"); t0 = time.time(); nb = 0; nrow = 0
for L, krs in mine:
    seq = [it for kr in krs for it in byrow[kr]]
    if all(it["key"] in have or (it["key"][0], it["key"][1], it["key"][2], it["key"][3]) in have for it in seq): continue
    nbt = len(seq); g = torch.tensor([gold[it["row"]] for it in seq], device=dev); ar = torch.arange(nbt, device=dev)
    S0 = {s: dict(rank=torch.zeros(nbt, dtype=torch.long, device=dev), top1=torch.zeros(nbt, dtype=torch.long, device=dev)) for s in (1, 2)}
    s0a = torch.zeros(nbt, STEPS, dtype=torch.long, device=dev); s0m = torch.zeros(nbt, STEPS, dtype=torch.bool, device=dev)
    def lh(step, x, logits):
        s0m[:, step] = x[:, -CANVAS] == m.mask_id; s0a[:, step] = torch.softmax(logits[:, -CANVAS], -1).max(-1).indices
        if step in S0:
            z = logits[:, -CANVAS].float(); gv = z[ar, g]; S0[step]["rank"] = (z > gv[:, None]).sum(-1); S0[step]["top1"] = z.argmax(-1)
        return logits
    fns = make_hooks(seq, L)
    for att in range(120):
        try: ans = m.generate([ids[it["row"]] for it in seq], fns, logits_hook=lh); break
        except (torch.OutOfMemoryError, RuntimeError) as _e:
            if "out of memory" not in str(_e): raise
            m.fns = {}; torch.cuda.empty_cache(); print("OOM, 60s 대기", flush=True); time.sleep(60)
    s0m_, s0a_ = s0m.cpu().numpy(), s0a.cpu().numpy(); S0c = {s: {k_: v.cpu().numpy() for k_, v in d.items()} for s, d in S0.items()}
    for j, it in enumerate(seq):
        lm_ = np.where(s0m_[j])[0]; last = int(lm_.max()) if len(lm_) else -1
        rec = dict(part=PART, role=it["role"], row=it["row"], donor=it["donor"], cond=it["cond"], pid=it.get("pid"), route=it.get("route"), fact=fid[it["row"]], k=int(k[it["row"]]),
                   answer=ans[j], new_type=m.grade(it["row"], ans[j]), gold_id=int(gold[it["row"]]), slot0_final=int(s0a_[j, last]) if last >= 0 else None, slot0_commit_step=last,
                   s1_rank=int(S0c[1]["rank"][j]), s1_top1=int(S0c[1]["top1"][j]), s2_rank=int(S0c[2]["rank"][j]), s2_top1=int(S0c[2]["top1"][j]), nseq=nbt)
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fo.flush(); nb += 1; nrow += nbt
    if nb % 10 == 0 or PIL: print(f"  묶음 {nb}/{len(mine)}, 열 {nrow}, {time.time() - t0:.0f}s ({(time.time() - t0) / nrow:.3f}s/열), 최대 메모리 {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
print(f"DONE {time.time() - t0:.0f}s", flush=True)
