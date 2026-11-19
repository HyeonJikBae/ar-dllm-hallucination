"""E46 (PREREG.md, CPU): 기존 결과를 AT-high (k 3,4) / AT-low (k 1,2) 로 나눠 재계산.  사실 (fid) 군집 부트스트랩 CI.
usage: python followup_mk/e46_at_split.py"""
import collections, glob, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import rows, k, fid, OUT, MODEL, REV
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
glen_c = {}
def glen(i):
    if i not in glen_c: glen_c[i] = len(tok(" " + rows[i]["gold_value"], add_special_tokens=False)["input_ids"])
    return glen_c[i]
m7 = {}
for l in open(f"{OUT}/e7/meta.jsonl"): d = json.loads(l); m7[d["row"]] = d["new_type"]
AT = lambda i: "AT-high" if k[i] in (3, 4) else ("AT-low" if k[i] in (1, 2) else None)
rng = np.random.default_rng(0); NB = 2000
def cboot(rows_, vals):
    """사실 군집 부트스트랩: 평균과 95% CI (단위 %)."""
    rows_ = np.asarray(rows_); vals = np.asarray(vals, float)
    if len(vals) == 0: return [None, None, None, 0]
    f = fid[rows_]; uf, inv = np.unique(f, return_inverse=True); s = np.bincount(inv, vals); n = np.bincount(inv)
    bs = []
    for _ in range(NB):
        c = np.bincount(rng.integers(0, len(uf), len(uf)), minlength=len(uf)); bs.append((c * s).sum() / max((c * n).sum(), 1))
    return [round(100 * float(vals.mean()), 1), round(100 * float(np.percentile(bs, 2.5)), 1), round(100 * float(np.percentile(bs, 97.5)), 1), int(len(vals))]
def cdiff(ga, gb):
    """서로 다른 사실 집합 (AT-high, AT-low) 사이 평균 차이의 CI: 각 군집 부트스트랩을 따로 뽑아 뺌."""
    out = []
    def one(rows_, vals):
        rows_ = np.asarray(rows_); vals = np.asarray(vals, float); f = fid[rows_]; uf, inv = np.unique(f, return_inverse=True); s = np.bincount(inv, vals); n = np.bincount(inv)
        c = np.bincount(rng.integers(0, len(uf), len(uf)), minlength=len(uf)); return (c * s).sum() / max((c * n).sum(), 1)
    for _ in range(NB): out.append(one(*ga) - one(*gb))
    m = np.mean(ga[1]) - np.mean(gb[1])
    return [round(100 * float(m), 1), round(100 * float(np.percentile(out, 2.5)), 1), round(100 * float(np.percentile(out, 97.5)), 1)]
def load(pattern, key="cond"):
    E = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{OUT}/{pattern}")):
        for l in open(f, encoding="utf-8"):
            try: d = json.loads(l)
            except Exception: continue
            E[d[key]][d["row"]] = d
    return E
O = {}
# ---------------- E20: 첫 토큰 고정
E20 = load("results_e20.jsonl"); o = {}
for cond, base, kind in (("정답토큰고정", "정답토큰고정|같은묶음base", "복구"), ("무작위토큰고정", "무작위토큰고정|같은묶음base", "복구"), ("오답토큰고정", "오답토큰고정|같은묶음base", "깨짐")):
    for at in ("AT-high", "AT-low"):
        rs = [r for r in E20[cond] if AT(r) == at and r in E20[base]]
        if kind == "복구":
            rs = [r for r in rs if E20[base][r]["new_type"] != "correct"]; v = [E20[cond][r]["new_type"] == "correct" for r in rs]
        else:
            rs = [r for r in rs if E20[base][r]["new_type"] == "correct"]; v = [E20[cond][r]["new_type"] != "correct" for r in rs]
        o[f"{cond} {kind} {at}"] = cboot(rs, v)
        if cond == "정답토큰고정":
            for L, nm in ((1, "1토큰"), (2, "2토큰"), (3, "3토큰+")):
                rr = [r for r in rs if (glen(r) if L < 3 else 3) == L or (L == 3 and glen(r) >= 3)]
                rr = [r for r in rs if (glen(r) == L if L < 3 else glen(r) >= 3)]
                o[f"{cond} 복구 {at} {nm}"] = cboot(rr, [E20[cond][r]["new_type"] == "correct" for r in rr])
# 정답 토큰 - 무작위 토큰 (짝 비교, 같은 행)
for at in ("AT-high", "AT-low"):
    rs = [r for r in E20["정답토큰고정"] if AT(r) == at and r in E20["무작위토큰고정"] and E20["정답토큰고정|같은묶음base"].get(r, {}).get("new_type") != "correct"]
    o[f"정답토큰-무작위토큰 복구 차이 {at}"] = cboot(rs, [float(E20["정답토큰고정"][r]["new_type"] == "correct") - float(E20["무작위토큰고정"][r]["new_type"] == "correct") for r in rs])
O["E20 첫 토큰 고정"] = o; print("E20", json.dumps(o, ensure_ascii=False), flush=True)
# ---------------- E37 검증 행 (붕괴·entity 행에 금 첫 토큰)
E37 = load("results_e37.jsonl"); o = {}
for at in ("AT-high", "AT-low"):
    rs = [r for r, d in E37["검증: 오답행 금토큰"].items() if AT(r) == at and m7.get(r) not in (None, "correct")]
    o[f"금 첫 토큰 고정 복구 {at} (E7 오답 행)"] = cboot(rs, [E37["검증: 오답행 금토큰"][r]["new_type"] == "correct" for r in rs])
    for L, nm in ((1, "1토큰"), (2, "2토큰"), (3, "3토큰+")):
        rr = [r for r in rs if (glen(r) == L if L < 3 else glen(r) >= 3)]
        o[f"금 첫 토큰 고정 복구 {at} {nm}"] = cboot(rr, [E37["검증: 오답행 금토큰"][r]["new_type"] == "correct" for r in rr])
zero = E37["금토큰고정"]
for L, nm in ((1, "1토큰"), (2, "2토큰"), (3, "3토큰+")):
    rr = [r for r in zero if (glen(r) == L if L < 3 else glen(r) >= 3)]
    o[f"(참고) 0/5 금 첫 토큰 고정 {nm}"] = cboot(rr, [zero[r]["new_type"] == "correct" for r in rr])
O["E37 금 첫 토큰 고정"] = o; print("E37", json.dumps(o, ensure_ascii=False), flush=True)
# ---------------- E32: τ 별 복구 / 정답 유지
E32 = load("results_e32.jsonl"); o = {}
for at in ("AT-high", "AT-low"):
    base = E32["정답토큰|base"]; rs = [r for r in base if AT(r) == at and base[r]["new_type"] != "correct"]
    for t in (0, 2, 4, 8, 16, 24):
        c = E32[f"정답토큰|tau{t}"]; rr = [r for r in rs if r in c]; o[f"정답토큰 τ{t} 복구 {at}"] = cboot(rr, [c[r]["new_type"] == "correct" for r in rr])
    b2 = E32["오답토큰|base"]; rs2 = [r for r in b2 if AT(r) == at and b2[r]["new_type"] == "correct"]
    for t in (0, 4, 8, 16):
        c = E32[f"오답토큰|tau{t}"]; rr = [r for r in rs2 if r in c]; o[f"오답토큰 τ{t} 정답 유지 {at}"] = cboot(rr, [c[r]["new_type"] == "correct" for r in rr])
O["E32 주입 시점"] = o; print("E32", json.dumps(o, ensure_ascii=False), flush=True)
# ---------------- E28 / E12 / E41: 모두 다시 생성 (첫 생성 E7)
E12 = load("results_e12_[A-Z].jsonl"); E41 = load("results_e41_[ABC].jsonl")
CON = {"층20 Brel c1": E12["Brel c1"], "층20 무작위 c1": E12["Rand c1"], "층20 다른관계 Brel c1": E12["BrelShuf c1"], "층20 B c0.5": E12["B c0.5"],
       "층20 Brel c0.5": E12.get("Brel c0.5", {}), "층24 B c0.5": E41["B c0.5"], "층24 B c1": E41["B c1"], "층24 Brel c1": E41["Brel c1"]}
o = {}; spec_vals = {}
for nm, E in CON.items():
    if not E: continue
    for at in ("AT-high", "AT-low"):
        w = [r for r in E if AT(r) == at and m7.get(r) not in (None, "correct")]; c = [r for r in E if AT(r) == at and m7.get(r) == "correct"]
        o[f"{nm} | 오답 복구 {at}"] = cboot(w, [E[r]["new_type"] == "correct" for r in w])
        for t in ("broken", "entity"):
            wt = [r for r in w if m7[r] == t]; o[f"{nm} | {t} 복구 {at}"] = cboot(wt, [E[r]["new_type"] == "correct" for r in wt])
        o[f"{nm} | 정답 깨짐 {at}"] = cboot(c, [E[r]["new_type"] != "correct" for r in c])
        allr = w + c; o[f"{nm} | 정답률 변화 {at}"] = cboot(allr, [float(E[r]["new_type"] == "correct") - float(m7[r] == "correct") for r in allr])
O["E28/E12/E41 모두 다시 생성"] = o
# 특이성: (방향 - 무작위) 복구 차이, 같은 행 짝.  그리고 AT-high 와 AT-low 의 차이
sp = {}
for nm in ("층20 Brel c1", "층20 다른관계 Brel c1", "층24 B c1", "층24 Brel c1"):
    E = CON[nm]; R_ = CON["층20 무작위 c1"]; g = {}
    for at in ("AT-high", "AT-low"):
        w = [r for r in E if AT(r) == at and m7.get(r) not in (None, "correct") and r in R_]
        vals = [float(E[r]["new_type"] == "correct") - float(R_[r]["new_type"] == "correct") for r in w]; g[at] = (w, vals)
        sp[f"{nm} - 무작위 (복구 차이) {at}"] = cboot(w, vals)
    sp[f"{nm} - 무작위: AT-high 대 AT-low 차이"] = cdiff(g["AT-high"], g["AT-low"])
O["특이성 (방향 - 무작위 복구)"] = sp; print("특이성", json.dumps(sp, ensure_ascii=False), flush=True)
# ---------------- E38: 첫 칸 금 토큰 순위
E38 = load("results_e38_[PQ].jsonl"); o = {}
allrank = [d["gold_rank_s0"] for d in E38["A"].values() if d.get("gold_rank_s0") is not None]; rmin = min(allrank); o["순위 최솟값 (1위 표기)"] = rmin
A = E38["A"]
for at in ("AT-high", "AT-low"):
    rs = [r for r in A if AT(r) == at and A[r]["new_type"] != "correct" and A[r].get("gold_rank_s0") is not None]
    o[f"base 1위 비율 {at}"] = cboot(rs, [A[r]["gold_rank_s0"] == rmin for r in rs]); o[f"base 중앙 순위 {at}"] = float(np.median([A[r]["gold_rank_s0"] for r in rs])) if rs else None
    for cnd in ("B", "C1", "D", "E1"):
        C = E38[cnd]; rr = [r for r in rs if r in C and C[r].get("gold_rank_s0") is not None]
        o[f"{cnd} 순위 상승 비율 {at}"] = cboot(rr, [C[r]["gold_rank_s0"] < A[r]["gold_rank_s0"] for r in rr])
        o[f"{cnd} 순위 하락 비율 {at}"] = cboot(rr, [C[r]["gold_rank_s0"] > A[r]["gold_rank_s0"] for r in rr])
        o[f"{cnd} 1위 비율 {at}"] = cboot(rr, [C[r]["gold_rank_s0"] == rmin for r in rr])
        o[f"{cnd} 복구 {at}"] = cboot(rr, [C[r]["new_type"] == "correct" for r in rr])
O["E38 첫 칸 금 토큰 순위 (base 오답 행, step 0)"] = o; print("E38", json.dumps(o, ensure_ascii=False), flush=True)
json.dump(O, open(f"{OUT}/e46_summary.json", "w"), ensure_ascii=False, indent=1)
# 보고서
L = ["# E46 AT-high (k 3,4) / AT-low (k 1,2) 재분석", "", "값: 평균 % [사실 군집 95% CI] (n).  첫 생성 E7, k 는 저장 라벨.", ""]
for sec, d in O.items():
    L.append(f"## {sec}"); L += [f"- {kx}: {v}" for kx, v in d.items()]; L.append("")
open(f"{OUT}/report_e46.md", "w").write("\n".join(L)); print("DONE", flush=True)
