"""E20 평가 (CPU).  usage: python followup_mk/e20_eval.py"""
import collections, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
OUT = "followup_mk"; rng = np.random.default_rng(0)
R = [json.loads(l) for l in open(f"{OUT}/results_e20.jsonl", encoding="utf-8")]
by = collections.defaultdict(dict); info = {}
for r in R: by[r["cond"]][r["row"]] = r["new_type"]; info[(r["cond"], r["row"])] = r
def rate(c, rows, pred): return np.array([pred(by[c][i]) for i in rows], float)
def ci(x, B=10000):
    ix = rng.integers(0, len(x), (B, len(x))); m = x[ix].mean(1) * 100; return [round(float(x.mean() * 100), 1), round(float(np.percentile(m, 2.5)), 1), round(float(np.percentile(m, 97.5)), 1)]
def dci(a, b, B=10000):
    d = a - b; ix = rng.integers(0, len(d), (B, len(d))); m = d[ix].mean(1) * 100; return [round(float(d.mean() * 100), 1), round(float(np.percentile(m, 2.5)), 1), round(float(np.percentile(m, 97.5)), 1)]
O = {}; sets = {i: r["set"] for c in by for i, r in ((i, info[(c, i)]) for i in by[c])}
def rows_of(c, st): return sorted(i for i in by[c] if sets[i] in st)
isc = lambda t: t == "correct"; isw = lambda t: t != "correct"
# 정답 토큰 고정 (붕괴, entity), 무작위 대조
for grp, st in (("붕괴+entity", ("broken", "entity")), ("붕괴", ("broken",)), ("entity", ("entity",))):
    c = "정답토큰고정"; rs = rows_of(c, st)
    if not rs: continue
    f = rate(c, rs, isc); b = rate(c + "|같은묶음base", rs, isc); rr = [i for i in rs if i in by["무작위토큰고정"]]
    o = {"n": len(rs), "고정 정답%": ci(f), "같은묶음 base 정답%": ci(b), "고정-base [CI]": dci(f, b)}
    if rr:
        fr = rate("무작위토큰고정", rr, isc); fb = rate(c, rr, isc); o.update({"n_무작위": len(rr), "무작위 토큰 정답%": ci(fr), "정답토큰-무작위 [CI]": dci(fb, fr)})
    O[f"H20.1 {grp}"] = o
# 오답 토큰 고정
for grp, st in (("정답(1~4/5)", ("correct",)), ("5/5", ("five",))):
    c = "오답토큰고정"; rs = rows_of(c, st)
    if not rs: continue
    f = rate(c, rs, isw); b = rate(c + "|같은묶음base", rs, isw)
    o = {"n": len(rs), "고정 후 오답%": ci(f), "같은묶음 base 오답%": ci(b), "고정-base [CI]": dci(f, b)}
    dt = collections.defaultdict(list)
    for i in rs: dt[info[(c, i)]["donor_type"]].append(isw(by[c][i]))
    o["공여 오답 유형별 오답%"] = {k_: [len(v), round(100 * float(np.mean(v)), 1)] for k_, v in dt.items()}
    o["결과 유형 분포"] = dict(collections.Counter(by[c][i] for i in rs)); o["같은 사실 공여%"] = round(100 * float(np.mean([info[(c, i)]["donor_same_fact"] for i in rs])), 1)
    O[f"H20.2 {grp}"] = o
# 정답 토큰 고정 실패 행: 무엇이 되었나
c = "정답토큰고정"; rs = rows_of(c, ("broken", "entity")); O["정답토큰고정 결과 유형"] = {st: dict(collections.Counter(by[c][i] for i in rows_of(c, (st,)))) for st in ("broken", "entity")}
O["공여 없어 빠진 행"] = "set_S 행 중 해당 조건 공여가 없는 행 수는 로그의 '공여 수' 참고"
h1 = O["H20.1 붕괴+entity"]; O["판정"] = {"H20.1": bool(h1["고정-base [CI]"][0] >= 40 and h1.get("정답토큰-무작위 [CI]", [0])[0] >= 30),
    "H20.2": bool(O["H20.2 정답(1~4/5)"]["고정 후 오답%"][0] >= 50 and O["H20.2 5/5"]["고정 후 오답%"][0] >= 20)}
json.dump(O, open(f"{OUT}/e20_summary.json", "w"), ensure_ascii=False, indent=1)
for k_, v in O.items(): print(k_, json.dumps(v, ensure_ascii=False))
