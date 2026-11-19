"""연결 교환 실험 채점 (PREREG_poison.md v3).  입력: results/{qwen,dream}_{data}.jsonl, data/musique_{data}.jsonl.
사용: python grade_bindswap.py [bindswap|bindswap_pilot]   출력: 화면 표, results/summary_{data}.json
"""
import collections, json, os, sys
import numpy as np
from grade import norm
from grade_poison_main import lab, M

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"


def boot_items(stat, n_items, n=10000, seed=0):
    rng = np.random.default_rng(seed); v = [stat(rng.integers(0, n_items, n_items)) for _ in range(n)]
    return [round(100 * stat(np.arange(n_items)), 1), round(100 * np.percentile(v, 2.5), 1), round(100 * np.percentile(v, 97.5), 1)]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bindswap"
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_{tag}.jsonl")}
    R = {m: {(r["qid"], r["cond"]): r for r in map(json.loads, open(f"{RES}/{m}_{tag}.jsonl"))} for m in M}
    L = lambda m, q, c, em=False: lab(R[m][(q, c)]["pred"], [data[q]["ans"][c]], data[q]["other"][c], em)
    out = {}
    for setname, Q in (("main", [q for q in data if all(L(m, q, "S") == "정답" for m in M)]), ("all", list(data))):
        o = {"n": len(Q), "S_acc": {m: round(100 * np.mean([L(m, q, "S") == "정답" for q in data]), 1) for m in M}}
        print(f"\n===== {setname}: {len(Q)} 문항  (S 정답률, 400 기준: {o['S_acc']}) =====")
        for em in (False, True):
            k = "em" if em else "f1"; o[k] = {}
            sel = {(m, c): np.array([np.mean([L(m, q, x, em) == "공격" for x in (c, c + "_sw")]) for q in Q]) for m in M for c in ("U", "L")}
            for c in ("U", "L"):
                d = {}
                for m in M:
                    lab2 = [[L(m, q, x, em) for x in (c, c + "_sw")] for q in Q]
                    d[m] = dict(정답=round(100 * np.mean([l.count("정답") / 2 for l in lab2]), 1), 다른값=round(100 * sel[m, c].mean(), 1),
                                연결추종=round(100 * np.mean([l == ["정답", "정답"] for l in lab2]), 1),
                                연결무시=round(100 * np.mean([norm(R[m][(q, c)]["pred"]) == norm(R[m][(q, c + "_sw")]["pred"]) for q in Q]), 1))
                d["dlm_adv"] = boot_items(lambda ix: sel["qwen", c][ix].mean() - sel["dream", c][ix].mean(), len(Q)); o[k][c] = d
                print(f"  [{k}] {c}: " + "  ".join(f"{m} {d[m]}" for m in M) + f"  DLM 우위 {d['dlm_adv']}")
            o[k]["J"] = boot_items(lambda ix: (sel["qwen", "U"][ix].mean() - sel["dream", "U"][ix].mean()) - (sel["qwen", "L"][ix].mean() - sel["dream", "L"][ix].mean()), len(Q))
            print(f"  [{k}] J = U 우위 - L 우위 = {o[k]['J']}")
        tq = collections.defaultdict(list)
        for i, q in enumerate(Q): tq[data[q]["template"]].append(q)
        sel = {(m, c): {q: np.mean([L(m, q, x) == "공격" for x in (c, c + "_sw")]) for q in Q} for m in M for c in ("U", "L")}
        o["by_template"] = {t: {c: round(100 * np.mean([sel["qwen", c][q] - sel["dream", c][q] for q in qs]), 1) for c in ("U", "L")} | {"n": len(qs)} for t, qs in sorted(tq.items())}
        print("  관계 틀별 DLM 우위 (U / L):", {t: (v["U"], v["L"], v["n"]) for t, v in o["by_template"].items()})
        out[setname] = o
    json.dump(out, open(f"{RES}/summary_{tag}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
