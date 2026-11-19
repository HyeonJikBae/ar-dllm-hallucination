"""연결 교환 (v3) 오류의 위치 요인 (PREREG_poison.md v8, 탐색적: 기술 통계는 이미 한 번 봤음).
칸 순서는 문항마다 무작위이고, 원래와 교환에서 정답 사실과 다른 사실의 역할이 뒤바뀐다.
  later  = 다른 값의 사실 문단이 정답 사실 문단보다 뒤 (질문에 더 가까움)
  nearer = 다른 값의 사실 문단이 대상 연결 문장 (A) 에 더 가까움 (칸 거리)
두 요인은 한 문항 안에서 함께 뒤집히므로, 두 요인이 엇갈리는 문항 (뒤쪽 사실이 연결 문장에서는 더 멂) 에서 따로 본다.
usage: python analyze_bind_errors.py     출력: results/summary_bind_errors.json
"""
import json, os
import numpy as np
from grade_poison_main import lab
from grade_bindswap import boot_items

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"


def main():
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_bindswap.jsonl")}
    R = {m: {(r["qid"], r["cond"]): r["pred"] for r in map(json.loads, open(f"{RES}/{m}_bindswap.jsonl"))} for m in ("qwen", "dream")}
    L = lambda m, q, c: lab(R[m][(q, c)], [data[q]["ans"][c]], data[q]["other"][c])
    Q = [q for q in data if all(L(m, q, "S") == "정답" for m in R)]
    S = {}
    for c in ("U", "L"):
        cases = []
        for q in Q:
            for cc in (c, c + "_sw"):
                ctx = data[q]["ctx"][cc]; la = ctx.index("LA" if cc == c else "LAs")
                ans_f, oth_f = ("FX", "FY") if cc == c else ("FY", "FX"); ia, io = ctx.index(ans_f), ctx.index(oth_f)
                cases.append(dict(q=q, later=io > ia, nearer=abs(io - la) < abs(ia - la), tie=abs(io - la) == abs(ia - la),
                                  err={m: L(m, q, cc) == "공격" for m in R}))
        out = {}
        by_q = {}
        for x in cases: by_q.setdefault(x["q"], []).append(x)
        for m in R:
            def within(key_a, key_b, sel):
                """문항 안 차이: (조건 a 인 경우의 오류) - (조건 b 인 경우의 오류).  sel 로 문항을 고름."""
                d = []
                for q, (u, v) in by_q.items():
                    if not sel(u): continue
                    ca, cb = (u, v) if key_a(u) else (v, u)
                    assert key_b(cb)
                    d.append(float(ca["err"][m]) - float(cb["err"][m]))
                d = np.array(d); return dict(n=len(d), diff=boot_items(lambda ix: d[ix].mean(), len(d)))
            e = np.array([x["err"][m] for x in cases], float)
            out[m] = dict(n_cases=len(e), err=round(100 * e.mean(), 1),
                          later_minus_earlier=within(lambda x: x["later"], lambda x: not x["later"], lambda x: True),
                          nearer_minus_farther=within(lambda x: x["nearer"], lambda x: not x["nearer"], lambda x: not x["tie"]),
                          엇갈림_later_farther_minus_earlier_nearer=within(lambda x: x["later"], lambda x: not x["later"],
                                                                           lambda x: not x["tie"] and x["later"] != x["nearer"]),
                          일치_later_nearer_minus_earlier_farther=within(lambda x: x["later"], lambda x: not x["later"],
                                                                         lambda x: not x["tie"] and x["later"] == x["nearer"]))
        S[c] = out; print(c, json.dumps(out, ensure_ascii=False, indent=None))
    json.dump(S, open(f"{RES}/summary_bind_errors.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
