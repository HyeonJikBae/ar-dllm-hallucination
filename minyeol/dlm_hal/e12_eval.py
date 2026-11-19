"""E12 평가 (PREREG.md, CPU).  첫 생성 = E7, 감지 점수 = '최종답과 다른 step 비율' (e11_twopass.py 와 같은 계산), 다시 생성 = results_e12_*.jsonl.
usage: python followup_mk/e12_eval.py"""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import rows, grp, k, fid, mid, NF, OUT, MODEL, REV, extract_answer, e7_rows
D7 = f"{OUT}/e7"; O = {}
_, _, F7 = e7_rows(); R = np.array([int(i) for i in np.where(mid)[0]] + [int(i) for i in F7]); n = len(R); ism = mid[R]; isf = ~ism
R7 = np.load(f"{D7}/rows.npy"); done = np.load(f"{D7}/done.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
p7 = {int(r): j for j, r in enumerate(R7)}; J = np.array([p7[i] for i in R]); assert all(done[J]) and all(int(i) in meta for i in R)
first = np.array([meta[int(i)]["new_type"] == "correct" for i in R])
# ---- 감지 점수 (e11 과 같은 계산)
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SPs = set(tok.all_special_ids)
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
x0 = lambda s: np.where(MK[:, s], IDS[:, s], committed)
def dc(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SPs), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
ANS = [[dc(r) for r in x0(s)] for s in range(32)]; fin = ANS[31]
O["궤적_최종답_E7답_일치"] = round(100 * float(np.mean([a == meta[int(i)]["answer"] for a, i in zip(fin, R)])), 2)
SC = np.array([np.mean([ANS[s][j] != fin[j] for s in range(31)]) for j in range(n)])      # 높을수록 '틀림'
print("궤적 복원 일치", O["궤적_최종답_E7답_일치"], flush=True)
# ---- 다시 생성 결과
E = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{OUT}/results_e12_[A-Z].jsonl")):
    for l in open(f, encoding="utf-8"):
        try: d = json.loads(l)
        except Exception: continue
        E[d["cond"]][d["row"]] = d["new_type"] == "correct"
CONDS = [c for c in ("B c0.5", "Brel c1", "ACE t27", "maskgit_plus", "entropy") if len(E[c]) == n]
if "--test" in sys.argv:   # 기계 점검: 다시 생성 = 첫 생성 / 무작위 뒤집기
    E["test_same"] = {int(i): bool(f_) for i, f_ in zip(R, first)}; E["test_rand"] = {int(i): bool(np.random.default_rng(1).random() < .5) for i in R}; CONDS = ["test_same", "test_rand"]
O["완료_조건"] = {c: len(E[c]) for c in E}; print(O["완료_조건"], flush=True)
g = grp[R]; fids = fid[R]
acc = lambda x: round(100 * float(x[ism].mean()), 2); acc5 = lambda x: round(100 * float(x[isf].mean()), 2)
def thresh_gate(sc, redo):
    """겹마다 다른 겹에서 1~4/5 정답률이 최대인 다시 생성 비율 q (평가 행 전체 기준 분위) 를 골라 적용."""
    sel = np.zeros(n, bool); qs = []
    for gg in range(NF):
        tr, te = g != gg, g == gg; bq, bv = 0.0, -1e9
        for q in np.linspace(0, 1, 51):
            th = np.quantile(sc[tr], 1 - q) if q > 0 else np.inf; v = np.where(sc >= th, redo, first)[tr & ism].mean()
            if v > bv + 1e-12: bv, bq = v, q
        th = np.quantile(sc[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = sc[te] >= th; qs.append(round(float(bq), 2))
    return sel, qs
rng = np.random.default_rng(0); B = 10000
mi, fi = np.where(ism)[0], np.where(isf)[0]
BI = rng.integers(0, len(mi), (B, len(mi)), dtype=np.int32)                       # 행 재표본 (1~4/5)
uf = np.unique(fids[mi]); fidx = {f: np.where(fids[mi] == f)[0] for f in uf}; BF = rng.integers(0, len(uf), (2000, len(uf)))   # 사실 묶음 재표본 (민감도)
def ci_gain(p):
    d = (p.astype(float) - first.astype(float))[mi]; bs = np.concatenate([d[BI[c0:c0 + 1000]].mean(1) for c0 in range(0, B, 1000)]) * 100
    bf = []
    for b in range(BF.shape[0]):
        ix = np.concatenate([fidx[uf[j]] for j in BF[b]]); bf.append(d[ix].mean() * 100)
    pval = (1 + np.sum(bs <= 0)) / (B + 1)
    return dict(이득=round(float(d.mean() * 100), 2), CI=[round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
                사실묶음CI=[round(float(np.percentile(bf, 2.5)), 2), round(float(np.percentile(bf, 97.5)), 2)], p한쪽=float(pval))
def summ(p, sel=None):
    out = {"1~4/5": acc(p), **ci_gain(p), "고침": round(100 * float((~first & p)[ism].mean()), 2), "깨짐": round(100 * float((first & ~p)[ism].mean()), 2), "5/5": acc5(p)}
    if sel is not None: out.update(다시생성_1to4비율=round(100 * float(sel[ism].mean()), 1), 다시생성_5of5비율=round(100 * float(sel[isf].mean()), 1))
    return out
O["첫생성"] = {"1~4/5": acc(first), "5/5": acc5(first)}; res = {}
for c in CONDS:
    redo = np.array([E[c][int(i)] for i in R]); r = {}
    sel, qs = thresh_gate(SC, redo); r["두번생성(겹밖문턱)"] = {**summ(np.where(sel, redo, first), sel), "q": qs}
    for q in (.3, .5):
        th = np.quantile(SC, 1 - q); s_ = SC >= th; r[f"두번생성(상위{int(q * 100)}%)"] = summ(np.where(s_, redo, first), s_)
    r["모두다시"] = summ(redo); r["오라클"] = {"1~4/5": acc(np.where(~first, redo, first)), "이득": round(acc(np.where(~first, redo, first)) - acc(first), 2)}
    res[c] = r; print(c, json.dumps(r, ensure_ascii=False), flush=True)
O["조건별"] = res
# Holm (두 번 생성 겹밖 문턱 이득 > 0, 한쪽)
ps = sorted([(res[c]["두번생성(겹밖문턱)"]["p한쪽"], c) for c in CONDS]); holm = {}; mx = 0
for j, (p, c) in enumerate(ps): mx = max(mx, min(1.0, (len(ps) - j) * p)); holm[c] = round(mx, 4)
O["Holm_p_두번생성"] = holm
# 같은 실행 base (민감도): E7 과 얼마나 같은가, 모두 다시 이득을 같은 실행 base 대비로
if len(E["base"]) == n:
    b12 = np.array([E["base"][int(i)] for i in R]); O["같은실행base"] = {"E7과_정오일치%": round(100 * float((b12 == first).mean()), 2), "1~4/5": acc(b12), "5/5": acc5(b12),
        "모두다시_이득_대_E12base": {c: round(acc(np.array([E[c][int(i)] for i in R])) - acc(b12), 2) for c in CONDS}}
    print("같은 실행 base", O["같은실행base"], flush=True)
if "B c0.5" in res:
    t = res["B c0.5"]["두번생성(겹밖문턱)"]; a = res["B c0.5"]["모두다시"]
    O["판정"] = {"H12.1": bool(t["이득"] >= 3 and t["CI"][0] > 0 and t["5/5"] >= 98), "H12.2": bool(a["CI"][0] > 0)}
print("판정", O.get("판정"), flush=True)
json.dump(O, open(f"{OUT}/e12_summary{'_test' if '--test' in sys.argv else ''}.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DONE", flush=True)
