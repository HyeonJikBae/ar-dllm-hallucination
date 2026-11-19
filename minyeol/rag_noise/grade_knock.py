"""knock_bind.py 채점 (PREREG_poison.md v7).  usage: python grade_knock.py     출력: results/summary_knock.json"""
import json, os
import numpy as np
from grade_poison_main import lab
from grade_bindswap import boot_items

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"; KO = ["none", "q", "ctrl"]


def main():
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_bindswap.jsonl")}
    K = {k: {(r["qid"], r["cond"]): r["pred"] for r in map(json.loads, open(f"{RES}/knock_dream_{k}.jsonl"))} for k in KO}
    data = {q: v for q, v in data.items() if all((q, "S") in K[k] for k in KO)}            # 실행이 덜 끝난 경우 대비
    old = {m: {(r["qid"], r["cond"]): r["pred"] for r in map(json.loads, open(f"{RES}/{m}_bindswap.jsonl"))} for m in ("qwen", "dream")}
    L = lambda P, q, c: lab(P[(q, c)], [data[q]["ans"][c]], data[q]["other"][c])
    main_v3 = [q for q in data if all(L(old[m], q, "S") == "정답" for m in old)]
    Q = [q for q in main_v3 if L(K["none"], q, "S") == "정답"]
    S = dict(n_v3=len(main_v3), n=len(Q), none_eq_gen=round(100 * np.mean([K["none"][k] == old["dream"][k] for k in K["none"]]), 1))
    S["S_acc"] = {k: round(100 * np.mean([L(K[k], q, "S") == "정답" for q in main_v3]), 1) for k in KO}
    print(f"v3 주 집합 {len(main_v3)}, 분석 {len(Q)}, none = gen.py 일치 {S['none_eq_gen']}%, S 정답률 {S['S_acc']}")
    for c in ("U", "L"):
        sel = {k: np.array([np.mean([L(K[k], q, x) == "공격" for x in (c, c + "_sw")]) for q in Q]) for k in KO}
        oth = {k: np.array([np.mean([L(K[k], q, x) == "그 밖" for x in (c, c + "_sw")]) for q in Q]) for k in KO}
        cor = {k: np.array([np.mean([L(K[k], q, x) == "정답" for x in (c, c + "_sw")]) for q in Q]) for k in KO}
        d = dict(다른값={k: round(100 * sel[k].mean(), 1) for k in KO}, 그밖={k: round(100 * oth[k].mean(), 1) for k in KO},
                 정답={k: round(100 * cor[k].mean(), 1) for k in KO},
                 Δq=boot_items(lambda ix: sel["q"][ix].mean() - sel["none"][ix].mean(), len(Q)),
                 Δctrl=boot_items(lambda ix: sel["ctrl"][ix].mean() - sel["none"][ix].mean(), len(Q)),
                 주_Δq_minus_Δctrl=boot_items(lambda ix: sel["q"][ix].mean() - sel["ctrl"][ix].mean(), len(Q)),
                 그밖_Δq_minus_Δctrl=boot_items(lambda ix: oth["q"][ix].mean() - oth["ctrl"][ix].mean(), len(Q)))
        S[c] = d; print(c, json.dumps(d, ensure_ascii=False))
    json.dump(S, open(f"{RES}/summary_knock.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
