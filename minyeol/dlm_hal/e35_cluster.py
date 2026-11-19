"""E35 (PREREG.md, CPU): 주요 CI 를 사실 (fid) 단위 군집 부트스트랩으로 다시 계산.  점추정은 각 실험 평가 스크립트와 같은 방식으로 재현하고, 행 단위 CI 와 나란히 낸다.
S 행 가중 실험 (E16, E19, E22) 은 set 별 층화 (set 안에서 사실을 복원 추출), 나머지는 사실을 복원 추출.  B = 4000."""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import grp, fid, mid, k, NF, OUT, e7_rows, set_S
B = 4000; O = {}
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); done = np.load(f"{D7}/done.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]; pos7 = {int(r): j for j, r in enumerate(RR)}
S1_all, SC_all = np.load(f"{OUT}/e30_scores.npy")                     # E7 행 순서 (RR)
first_of = lambda i: meta[int(i)]["new_type"] == "correct"

def load_cond(files):
    E = collections.defaultdict(dict); SET = {}
    for f in files:
        for l in open(f, encoding="utf-8"):
            try: d = json.loads(l)
            except Exception: continue
            E[d["cond"]][d["row"]] = d["new_type"] == "correct"; SET[d["row"]] = d.get("set")
    return E, SET

# ---------- 공용 부트스트랩
def idx_cluster(groups, strata, rng):
    """strata 안에서 groups (사실) 를 복원 추출한 행 인덱스 목록 B 개를 만드는 생성기."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for j, (s, g) in enumerate(zip(strata, groups)): by[s][g].append(j)
    keys = {s: list(v.keys()) for s, v in by.items()}
    for _ in range(B):
        out = []
        for s, ks in keys.items():
            pick = rng.integers(0, len(ks), len(ks))
            for p in pick: out.extend(by[s][ks[p]])
        yield np.array(out)
def idx_row(strata, rng):
    by = collections.defaultdict(list)
    for j, s in enumerate(strata): by[s].append(j)
    by = {s: np.array(v) for s, v in by.items()}
    for _ in range(B): yield np.concatenate([v[rng.integers(0, len(v), len(v))] for v in by.values()])
def ci_of(stat, n, groups, strata, seed=0):
    """stat(ix) -> 값.  행 단위 / 군집 단위 95% CI."""
    rng = np.random.default_rng(seed); r = np.array([stat(ix) for ix in idx_row(strata, rng)])
    rng = np.random.default_rng(seed); c = np.array([stat(ix) for ix in idx_cluster(groups, strata, rng)])
    q = lambda a: [round(float(np.percentile(a, 2.5)), 2), round(float(np.percentile(a, 97.5)), 2)]
    return q(r), q(c)
def entry(point, reported, rci, cci):
    return {"점추정": round(float(point), 2), "보고값": reported, "재현": (None if reported is None else bool(abs(round(float(point), 2) - reported) <= 0.02)),
            "행 CI": rci, "군집 CI": cci, "하한>0 (행)": bool(rci[0] > 0), "하한>0 (군집)": bool(cci[0] > 0), "0 포함 여부 바뀜": bool((rci[0] > 0) != (cci[0] > 0) or (rci[1] < 0) != (cci[1] < 0))}

# ---------- S 행 가중 실험 (E16, E19, E22): wacc = set 가중
WS = {"broken": 310 / 310, "entity": 2749 / 300, "correct": 2875 / 300, "zero": 44110 / 300, "five": 5165 / 300}
def s_block(E, SET, conds, reported, two_pass_conds=(), rep_tp=None, seed=0):
    S = [i for i in E["base"]]; sets = np.array([SET[i] for i in S]); ws = np.array([WS[x] for x in sets]); mS = np.isin(sets, ["broken", "entity", "correct"])
    gS = grp[S]; fS = fid[S]; base = np.array([E["base"][i] for i in S]); first = np.array([first_of(i) for i in S]); SC = np.array([SC_all[pos7[int(i)]] for i in S])
    def wacc(x, ix): m = mS[ix]; w = ws[ix]; return 100 * (x[ix] * w)[m].sum() / w[m].sum()
    out = {}
    for c in conds:
        x = np.array([E[c][i] for i in S]); f = lambda ix, x=x: wacc(x, ix) - wacc(base, ix); allix = np.arange(len(S))
        r, cc = ci_of(f, len(S), fS, sets, seed); out[f"{c} 순이득"] = entry(f(allix), reported.get(c), r, cc)
    def twopass(redo):
        sel = np.zeros(len(S), bool); allix = np.arange(len(S))
        for g in range(NF):
            tr, te = gS != g, gS == g; bq, bv = 0.0, -1e9
            for q in np.linspace(0, 1, 51):
                th = np.quantile(SC[tr], 1 - q) if q > 0 else np.inf; v = wacc(np.where(SC >= th, redo, first), np.where(tr)[0])
                if v > bv: bv, bq = v, q
            th = np.quantile(SC[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = SC[te] >= th
        return np.where(sel, redo, first)
    for c in two_pass_conds:
        p = twopass(np.array([E[c][i] for i in S])); f = lambda ix, p=p: wacc(p, ix) - wacc(first, ix)
        r, cc = ci_of(f, len(S), fS, sets, seed); out[f"{c} 두 번 생성"] = entry(f(np.arange(len(S))), (rep_tp or {}).get(c), r, cc)
    return out

e16s = json.load(open(f"{OUT}/e16_summary.json"))["조건별"]
E, SET = load_cond(sorted(f for f in glob.glob(f"{OUT}/results_e16_*.jsonl") if "_pilot" not in f and "_full" not in f))
c16 = ["Brel c1", "B c0.5", "정적+동적 B c2", "무작위 B c2", "정적+동적 Brel c2", "변형 Brel c1", "변형 Brel c2"]
O["E16"] = s_block(E, SET, c16, {c: e16s[c]["순이득"] for c in c16}, two_pass_conds=["Brel c1", "변형 Brel c2"], rep_tp={c: e16s[c]["두번생성"]["이득"] for c in ("Brel c1", "변형 Brel c2")})
# H16.2 차이 (정적+동적 B c2 - 무작위 B c2)
S = [i for i in E["base"]]; sets = np.array([SET[i] for i in S]); ws = np.array([WS[x] for x in sets]); mS = np.isin(sets, ["broken", "entity", "correct"])
def wacc_(x, ix): m = mS[ix]; w = ws[ix]; return 100 * (x[ix] * w)[m].sum() / w[m].sum()
xa = np.array([E["정적+동적 B c2"][i] for i in S]); xb = np.array([E["무작위 B c2"][i] for i in S]); f = lambda ix: wacc_(xa, ix) - wacc_(xb, ix)
r, cc = ci_of(f, len(S), fid[S], sets); O["E16"]["H16.2 정적+동적 B c2 - 무작위 B c2"] = entry(f(np.arange(len(S))), 0.07, r, cc)
print("E16", json.dumps(O["E16"], ensure_ascii=False)[:1500], flush=True)

e19s = json.load(open(f"{OUT}/e19_summary.json"))["조건별"]
E, SET = load_cond(sorted(f for f in glob.glob(f"{OUT}/results_e19_*.jsonl") if "_pilot" not in f))
c19 = ["LDA_mid_0.9 c0.5", "LDA_mid_0.9 c1", "LDA_mid_0.9 c2", "LDA_전체_0.1 c0.5", "LDA_전체_0.1 c1"]
O["E19"] = s_block(E, SET, c19, {c: e19s[c]["순이득"] for c in c19}, two_pass_conds=["LDA_mid_0.9 c1"], rep_tp={"LDA_mid_0.9 c1": e19s["LDA_mid_0.9 c1"]["두번생성"]["이득"]})
print("E19", json.dumps(O["E19"], ensure_ascii=False)[:800], flush=True)

e22s = json.load(open(f"{OUT}/e22_summary.json"))
E, SET = load_cond([f"{OUT}/results_e22.jsonl"])
c22 = ["ar", "ar + Brel c1 [사후]", "orig + Brel c1 [사후]"]
O["E22"] = s_block(E, SET, c22, {c: e22s[c]["이득[CI]"][0] for c in c22})
print("E22", json.dumps(O["E22"], ensure_ascii=False)[:800], flush=True)

# ---------- E20 (행 단위 비가중, 같은 묶음 base 와 짝)
e20s = json.load(open(f"{OUT}/e20_summary.json")); E, SET = load_cond([f"{OUT}/results_e20.jsonl"]); O["E20"] = {}
def paired(ca, cb, sets_keep, reported, flip=False, name=""):
    rows_ = [i for i in E[ca] if SET[i] in sets_keep and i in E[cb]]; a = np.array([E[ca][i] for i in rows_], float); b = np.array([E[cb][i] for i in rows_], float)
    if flip: a, b = 1 - a, 1 - b
    d = 100 * (a - b); f = lambda ix: d[ix].mean(); r, cc = ci_of(f, len(d), fid[rows_], np.zeros(len(d))); O["E20"][name] = {**entry(f(np.arange(len(d))), reported, r, cc), "n": len(d)}
paired("정답토큰고정", "정답토큰고정|같은묶음base", ("broken", "entity"), e20s["H20.1 붕괴+entity"]["고정-base [CI]"][0], name="정답 첫 토큰 고정 - base (붕괴+entity)")
paired("정답토큰고정", "무작위토큰고정", ("broken", "entity"), e20s["H20.1 붕괴+entity"]["정답토큰-무작위 [CI]"][0], name="정답 토큰 - 무작위 토큰")
paired("오답토큰고정", "오답토큰고정|같은묶음base", ("correct",), e20s["H20.2 정답(1~4/5)"]["고정-base [CI]"][0], flip=True, name="오답 고정 후 오답 - base (1~4/5 정답 행)")
paired("오답토큰고정", "오답토큰고정|같은묶음base", ("five",), e20s["H20.2 5/5"]["고정-base [CI]"][0], flip=True, name="오답 고정 후 오답 - base (5/5)")
print("E20", json.dumps(O["E20"], ensure_ascii=False)[:800], flush=True)

# ---------- E21 (복구율, 비 b/c)
e21s = json.load(open(f"{OUT}/e21_summary.json")); E, SET = load_cond([f"{OUT}/results_e21.jsonl"]); O["E21"] = {}
rows21 = [i for i in E["c 전체 | Brel c1"]]; f21 = fid[rows21]; z = np.zeros(len(rows21))
for c in E:
    x = np.array([E[c][i] for i in rows21], float) * 100; f = lambda ix, x=x: x[ix].mean(); r, cc = ci_of(f, len(x), f21, z); O["E21"][c] = entry(f(np.arange(len(x))), e21s["조건별"][c]["전체"]["복구"], r, cc)
xb = np.array([E["b 답구간 | Brel c1"][i] for i in rows21], float); xc = np.array([E["c 전체 | Brel c1"][i] for i in rows21], float)
f = lambda ix: xb[ix].mean() / max(xc[ix].mean(), 1e-9); r, cc = ci_of(f, len(xb), f21, z)
rr = lambda a: [round(v, 3) for v in a]; O["E21"]["H21.1 비 (b+Brel)/(c+Brel)"] = {"점추정": round(float(f(np.arange(len(xb)))), 3), "보고값": e21s["H21.1"]["비"], "행 CI": r, "군집 CI": cc, "하한>=0.7 (행)": bool(r[0] >= .7), "하한>=0.7 (군집)": bool(cc[0] >= .7)}
print("E21", json.dumps(O["E21"], ensure_ascii=False)[:800], flush=True)

# ---------- 전체 규모 (E25, E26, E30): R = 1~4/5 5,950 + 5/5 2,000
_, _, F7 = e7_rows(); R = np.array([int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]); ismR = mid[R]; fR = fid[R]; gR = grp[R]
firstR = np.array([first_of(i) for i in R]); SCR = np.array([SC_all[pos7[int(i)]] for i in R])
def loadR(pat, cond):
    E_ = {}
    for f in sorted(glob.glob(f"{OUT}/{pat}")):
        for l in open(f, encoding="utf-8"):
            try: d = json.loads(l)
            except Exception: continue
            if d["cond"] == cond: E_[d["row"]] = d["new_type"] == "correct"
    return np.array([E_[int(i)] for i in R])
dense = loadR("results_e12_[A-Z].jsonl", "Brel c1"); var = loadR("results_e16_full_G.jsonl", "변형 Brel c1")
mi = np.where(ismR)[0]; fmi = fR[mi]; zmi = np.zeros(len(mi))
def mid_gain(p): d = 100 * (p.astype(float) - firstR.astype(float))[mi]; return d
def tp_select(redo):
    sel = np.zeros(len(R), bool)
    for gg in range(NF):
        tr, te = gR != gg, gR == gg; bq, bv = 0.0, -1e9
        for q in np.linspace(0, 1, 51):
            th = np.quantile(SCR[tr], 1 - q) if q > 0 else np.inf; v = np.where(SCR >= th, redo, firstR)[tr & ismR].mean()
            if v > bv + 1e-12: bv, bq = v, q
        th = np.quantile(SCR[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = SCR[te] >= th
    return np.where(sel, redo, firstR)
e25s = json.load(open(f"{OUT}/e25_summary.json")); O["E25"] = {}
for nm, p, rep in (("변형 한 번 생성", var, e25s["한 번 생성 (변형 Brel c1)"]["1~4/5 이득 [CI]"][0]), ("조밀 한 번 생성", dense, e25s["한 번 생성 (조밀 Brel c1)"]["1~4/5 이득 [CI]"][0]),
                   ("변형 두 번 생성", tp_select(var), e25s["두 번 생성 (변형)"]["1~4/5 이득 [CI]"][0]), ("조밀 두 번 생성 (E12)", tp_select(dense), e25s["두 번 생성 (조밀)"]["1~4/5 이득 [CI]"][0])):
    d = mid_gain(p); f = lambda ix, d=d: d[ix].mean(); r, cc = ci_of(f, len(d), fmi, zmi); O["E25"][nm] = entry(f(np.arange(len(d))), rep, r, cc)
d = 100 * (var.astype(float) - dense.astype(float))[mi]; f = lambda ix: d[ix].mean(); r, cc = ci_of(f, len(d), fmi, zmi); O["E25"]["변형 - 조밀 (한 번 생성)"] = entry(f(np.arange(len(d))), e25s["변형 - 조밀 (1~4/5)"][0], r, cc)
print("E25", json.dumps(O["E25"], ensure_ascii=False)[:900], flush=True)

# E26: 조합 - SC (같은 비율).  첫 칸 신호 재계산 (e26_first_slot.py 와 같은 정의)
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); MG = np.asarray(np.load(f"{D7}/mg.npy", mmap_mode="r")[J]).astype(np.float32); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J]); n7 = len(RR)
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); S1f = -MG[:, 0, 0]; S2f = lastm[:, 0].astype(float)
m0 = MK[:, :, 0]; a0 = IDS[:, :, 0]; S3f = np.zeros(n7)
for j in range(n7):
    st_ = np.where(m0[j])[0]; seq = a0[j, st_]; S3f[j] = (seq[1:] != seq[:-1]).sum() if len(seq) > 1 else 0
rk = lambda x: (np.argsort(np.argsort(x)) + 0.5) / len(x)
C = {"C1 SC+S2": rk(SC_all) + rk(S2f), "C2 SC+S1": rk(SC_all) + rk(S1f), "C3 SC+S3": rk(SC_all) + rk(S3f)}
e26s = json.load(open(f"{OUT}/e26_summary.json"))["조합 - SC 이득 차이"]; O["E26"] = {}
ixR = np.array([pos7[int(i)] for i in R])
def sel_q(s, q): sR = s[ixR]; return np.where(sR >= np.quantile(sR, 1 - q), dense, firstR)
for nm, s in C.items():
    for q in (.4, .5):
        d = mid_gain(sel_q(s, q)) - mid_gain(sel_q(SC_all, q)); f = lambda ix, d=d: d[ix].mean(); r, cc = ci_of(f, len(d), fmi, zmi)
        O["E26"][f"{nm} 상위{int(q * 100)}% - SC"] = entry(f(np.arange(len(d))), e26s.get(f"{nm} 상위{int(q * 100)}% - SC", [None])[0], r, cc)
print("E26", json.dumps(O["E26"], ensure_ascii=False)[:900], flush=True)

# E30: 결합 - SC (예산 20%), 결합 1~4/5 이득.  E7 전 행, 모집단 가중 정책, 1~4/5 이득은 1~4/5 행 평균.
kk = k[RR]; isz7 = kk == 0; isf7 = kk == 5; ism7 = (kk >= 1) & (kk <= 4); corr7 = np.array([first_of(i) for i in RR])
W7 = np.where(isz7, 44110 / isz7.sum(), np.where(isf7, 5165 / isf7.sum(), 1.0))
def wrank(x):
    o = np.argsort(x); cw = np.cumsum(W7[o]); r = np.empty(len(x)); r[o] = (cw - W7[o] / 2) / W7.sum(); return r
COMB = wrank(S1_all) * wrank(SC_all)
E12 = {}
for f in sorted(glob.glob(f"{OUT}/results_e12_[A-Z].jsonl")):
    for l in open(f, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        if d["cond"] == "Brel c1": E12[d["row"]] = d["new_type"] == "correct"
redo7 = np.array([E12.get(int(i), c) if not z else c for i, c, z in zip(RR, corr7, isz7)])
def policy(s, b):
    o = np.argsort(-s); cw = np.cumsum(W7[o]); sel = np.zeros(len(s), bool); sel[o[cw <= b * W7.sum()]] = True; return np.where(sel, redo7, corr7)
m7 = np.where(ism7)[0]; f7 = fid[RR][m7]; z7 = np.zeros(len(m7)); e30s = json.load(open(f"{OUT}/e30_summary.json")); O["E30"] = {}
for b in (.1, .2):
    dC = 100 * (policy(COMB, b).astype(float) - corr7.astype(float))[m7]; dS = 100 * (policy(SC_all, b).astype(float) - corr7.astype(float))[m7]
    for nm, d, rep in ((f"결합 1~4/5 이득 (예산 {int(b * 100)}%)", dC, e30s[f"예산 {int(b * 100)}%"]["결합"]["1~4/5 이득"]), (f"결합 - SC (예산 {int(b * 100)}%)", dC - dS, e30s[f"예산 {int(b * 100)}%"]["결합-SC 1~4/5 이득 차이 [CI]"][0])):
        f = lambda ix, d=d: d[ix].mean(); r, cc = ci_of(f, len(d), f7, z7); O["E30"][nm] = entry(f(np.arange(len(d))), rep, r, cc)
print("E30", json.dumps(O["E30"], ensure_ascii=False)[:900], flush=True)

# ---------- 요약: 0 포함 여부가 바뀐 항목
O["0 포함 여부가 바뀐 항목"] = [f"{ex}: {nm}" for ex, v in O.items() if isinstance(v, dict) for nm, e in v.items() if isinstance(e, dict) and e.get("0 포함 여부 바뀜")]
O["재현 안 된 점추정"] = [f"{ex}: {nm} (재현 {e['점추정']} 대 보고 {e['보고값']})" for ex, v in O.items() if isinstance(v, dict) for nm, e in v.items() if isinstance(e, dict) and e.get("재현") is False]
json.dump(O, open(f"{OUT}/e35_summary.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("바뀐 항목", O["0 포함 여부가 바뀐 항목"]); print("재현 안 됨", O["재현 안 된 점추정"]); print("DONE", flush=True)
