"""다른 모델 쌍 채점 (PREREG_poison.md v14).  usage: python grade_pair2.py [layout|natural]     출력: results/summary_pair2_{layout,natural}.json"""
import glob, json, os, sys
import numpy as np
from grade import f1
from grade_layout import load_data, load_pred, L, s_ok, analyze, cube, slope8, boot, RES, HERE
from gen_fmt import select


def layout():
    data = load_data("layout"); sub = [r["qid"] for r in select(list(data.values()), 2)]; S = {}
    P = {m: load_pred(m, "layout", "base") for m in ("llama2", "diffullama", "lladamoe", "qwen", "dream")}
    has = lambda m: [q for q in sub if all((q, c) in P[m] for c in data[q]["ctx"] if c[0] in "LS")]
    conds = lambda *ms: [c for c in ("U", "L") if all((q0, f"{c}1") in P[m] for m in ms for q0 in has(ms[0])[:1])]
    for m in P:
        qs = has(m); S.setdefault("S_acc", {})[m] = round(100 * np.mean([L(P[m], data, q, c) == "정답" for q in qs for c in ("S_e", "S_l")]), 1) if qs else None
    print("S 정답률 (200 문항):", S["S_acc"])
    for name, a, b in (("llama2-diffullama", "llama2", "diffullama"), ("qwen-dream", "qwen", "dream")):
        Q = [q for q in sub if q in set(has(a)) & set(has(b)) and s_ok(P[a], data, q) and s_ok(P[b], data, q)]; S[name] = dict(n=len(Q))
        if not Q: continue
        for c in conds(a, b):
            da, Ya = analyze(P[a], data, Q, c); db, Yb = analyze(P[b], data, Q, c)
            S[name][c] = {a: da, b: db, "다른값_차": boot(Ya.mean((1, 2)) - Yb.mean((1, 2))), "기울기x8_차": boot(slope8(Ya) - slope8(Yb))}
            print(f"[{name} {c}] n {len(Q)}  {a} 다른값 {da['다른값']} 기울기 {da['기울기x8']} x별 {da['x별']}")
            print(f"    {b} 다른값 {db['다른값']} 기울기 {db['기울기x8']} x별 {db['x별']}")
            print(f"    차이: 다른값 {S[name][c]['다른값_차']}  기울기 {S[name][c]['기울기x8_차']}")
    Q = [q for q in has("lladamoe") if s_ok(P["lladamoe"], data, q)]; S["lladamoe"] = dict(n=len(Q))
    if Q:
        for c in conds("lladamoe"):
            d, _ = analyze(P["lladamoe"], data, Q, c); S["lladamoe"][c] = d; print(f"[lladamoe {c}] n {len(Q)} 다른값 {d['다른값']} 그밖 {d['그밖']} 기울기 {d['기울기x8']} x별 {d['x별']}")
    json.dump(S, open(f"{RES}/summary_pair2_layout.json", "w"), ensure_ascii=False, indent=1)


def natural():
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_anchor.jsonl")}
    def load(n): return {(r["qid"], r["cond"]): r["pred"] for p in glob.glob(f"{RES}/{n}.jsonl") + glob.glob(f"{RES}/{n}.sh*.jsonl") for r in map(json.loads, open(p))}
    P = {"llama2": load("llama2_anchor_base"), "diffullama": load("diffullama_anchor_base"), "lladamoe": load("lladamoe_anchor_base"), "qwen": load("qwen_anchorbs1"), "dream": load("dream_anchorbs1_s16")}
    S = {}; Q = [q for q in data if all((q, c) in P[m] for m in P for c in ("gold", "hard3"))]; S["n"] = len(Q)
    F = lambda m, c: np.array([max(f1(P[m][(q, c)], a) for a in data[q]["answers"]) for q in Q]); D = {m: F(m, "gold") - F(m, "hard3") for m in P}
    S["F1"] = {m: dict(gold=round(100 * F(m, "gold").mean(), 1), hard3=round(100 * F(m, "hard3").mean(), 1), 피해=boot(D[m])) for m in P}
    S["격차"] = {"llama2-diffullama": boot(D["llama2"] - D["diffullama"]), "qwen-dream": boot(D["qwen"] - D["dream"]), "qwen-lladamoe": boot(D["qwen"] - D["lladamoe"]), "llama2-lladamoe": boot(D["llama2"] - D["lladamoe"])}
    print(json.dumps(S, ensure_ascii=False, indent=1)); json.dump(S, open(f"{RES}/summary_pair2_natural.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    natural() if "natural" in sys.argv else layout()
