"""E21 평가 (PREREG.md, CPU).  조건별 복구율 (첫 생성이 오답인 행 -> 정답), 첫 생성 유형 (붕괴 / entity) 별, 부트스트랩 CI, H21.1, H21.2.
usage: python followup_mk/e21_eval.py"""
import collections, json
import numpy as np
R = collections.defaultdict(dict)
for l in open("followup_mk/results_e21.jsonl", encoding="utf-8"):
    d = json.loads(l); R[d["cond"]][d["row"]] = d
CONDS = ["a 칸0 | 없음", "a 칸0 | Brel c1", "b 답구간 | 없음", "b 답구간 | Brel c1", "c 전체 | 없음", "c 전체 | Brel c1"]
rows = sorted(set.intersection(*[set(R[c]) for c in CONDS])); n = len(rows)
ty = {i: R["c 전체 | 없음"][i]["first_type"] for i in rows}
rng = np.random.default_rng(0); BIX = rng.integers(0, n, (10000, n))
def rate(c, sel=None):
    ix = [j for j, i in enumerate(rows) if sel is None or ty[i] in sel]; v = np.array([R[c][rows[j]]["new_type"] == "correct" for j in ix], float)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)]) * 100
    return dict(n=len(v), 복구=round(100 * float(v.mean()), 1), CI=[round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)])
O = {"행 수": n, "첫생성 유형": dict(collections.Counter(ty.values()))}
O["조건별"] = {c: {"전체": rate(c), "붕괴": rate(c, ("broken",)), "entity": rate(c, ("entity",))} for c in CONDS}
O["같은 답 비율 (첫 생성과)"] = {c: round(100 * float(np.mean([R[c][i]["answer"] == R[c][i]["first_answer"] for i in rows])), 1) for c in CONDS}
# H21.1: (b)+Brel 이 (c)+Brel 의 70% 이상 (복구율 비) + 잡음 바닥 (c 없음) 을 뺀 값도 함께
fl = O["조건별"]["c 전체 | 없음"]["전체"]["복구"]; b_ = O["조건별"]["b 답구간 | Brel c1"]["전체"]["복구"]; c_ = O["조건별"]["c 전체 | Brel c1"]["전체"]["복구"]
rb = np.array([R["b 답구간 | Brel c1"][i]["new_type"] == "correct" for i in rows], float); rc = np.array([R["c 전체 | Brel c1"][i]["new_type"] == "correct" for i in rows], float)
ratio = (rb[BIX].mean(1)) / np.maximum(rc[BIX].mean(1), 1e-9)
O["H21.1"] = dict(b_Brel=b_, c_Brel=c_, 비=round(b_ / c_, 3) if c_ > 0 else None, 비_CI=[round(float(np.percentile(ratio, 2.5)), 3), round(float(np.percentile(ratio, 97.5)), 3)], 잡음바닥_c없음=fl,
                  통과=bool(c_ > 0 and b_ / c_ >= 0.7))
a0, b0 = O["조건별"]["a 칸0 | 없음"]["전체"]["복구"], O["조건별"]["b 답구간 | 없음"]["전체"]["복구"]
O["H21.2"] = dict(a_없음=a0, b_없음=b0, 통과=bool(a0 <= 5 and b0 <= 5))
O["해석 메모"] = "복구율 = 첫 생성이 오답인 행 중 다시 디코딩 뒤 정답인 비율.  c 없음 은 처음부터 다시 생성이라 잡음 바닥 (배치 구성 차이)."
json.dump(O, open("followup_mk/e21_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k_: v for k_, v in O.items() if k_ != "조건별"}, ensure_ascii=False))
for c in CONDS: print(f"{c:20s} 전체 {O['조건별'][c]['전체']}  붕괴 {O['조건별'][c]['붕괴']['복구']} (n={O['조건별'][c]['붕괴']['n']})  entity {O['조건별'][c]['entity']['복구']} (n={O['조건별'][c]['entity']['n']})")
