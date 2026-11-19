"""Dream 생성 조건 비교 채점 (PREREG_poison.md v13).  usage: python grade_gen.py [--natural]     출력: results/summary_gen_{layout,natural}.json"""
import glob, json, os, sys
import numpy as np
from grade import f1, norm
from grade_layout import load_data, load_pred, L, s_ok, cube, slope8, boot, M, RES, HERE
from gen_fmt import select

ARMS = ["l2r_m16", "ent_s4", "ent_s1", "l2r_m8", "l2r_m64"]


def eq_rule(b):
    """차이 CI 판정: [-5, 5] 안이면 차이 없음, 하한이 0 위면 증가, 상한이 0 아래면 감소."""
    return "차이 없음" if b[1] >= -5 and b[2] <= 5 else "증가" if b[1] > 0 else "감소" if b[2] < 0 else "판단 보류"


def first_word_cube(P, data, Q, c):
    """첫 단어 라벨: 다른 값의 첫 단어와 같으면 1, 첫 단어가 같은 값 쌍은 제외 (nan)."""
    Y = np.full((len(Q), 5, 2), np.nan)
    from grade_layout import KS
    for i, q in enumerate(Q):
        for j, k in enumerate(KS):
            for s in ("", "_sw"):
                cc = f"{c}{k}{s}"; role = 0 if data[q]["meta"][cc]["oth_slot"] == 6 else 1
                w = norm(P[(q, cc)]).split(); a, o = norm(data[q]["ans"][cc]).split()[0], norm(data[q]["other"][cc]).split()[0]
                if a != o: Y[i, j, role] = float(bool(w) and w[0] == o)
    return Y


def layout():
    data = load_data("layout"); Pq = load_pred("qwen", "layout", "base"); Pd = load_pred("dream", "layout", "base")
    Q0 = [q for q in data if all(s_ok(P, data, q) for P in (Pq, Pd))]; sub = {r["qid"] for r in select(list(data.values()), 2)}; Q = [q for q in Q0 if q in sub]
    arms = [a for a in ARMS if glob.glob(f"{RES}/dream_layout_{a}*.jsonl")]; S = dict(n_v10=len(Q0), n=len(Q), arms=arms); print(f"v10 주 집합 {len(Q0)}, 200 문항 안 {len(Q)}")
    Yq = cube(Pq, data, Q, "L"); Yb = cube(Pd, data, Q, "L"); item = lambda A: A.mean((1, 2))
    Gb = item(Yq) - item(Yb); S["base"] = dict(다른값=round(100 * Yb.mean(), 1), 그밖=round(100 * cube(Pd, data, Q, "L", "그 밖").mean(), 1), 기울기x8=boot(slope8(Yb)), G=boot(Gb), 첫단어=round(100 * np.nanmean(first_word_cube(Pd, data, Q, "L")), 1))
    S["qwen"] = dict(다른값=round(100 * Yq.mean(), 1), 기울기x8=boot(slope8(Yq))); print("Qwen base", S["qwen"]); print("Dream base", S["base"])
    for a in arms:
        P = load_pred("dream", "layout", a); Q2 = [q for q in Q if all((q, c) in P for c in data[q]["ctx"] if c[0] in "LS")]
        ix = [Q.index(q) for q in Q2]; Y = cube(P, data, Q2, "L"); d = dict(n=len(Q2))
        d["S_acc"] = round(100 * np.mean([L(P, data, q, c) == "정답" for q in Q2 for c in ("S_e", "S_l")]), 1); d["관문"] = "통과" if d["S_acc"] >= 80 else "미달 (기술 통계만)"
        d["다른값"] = round(100 * Y.mean(), 1); d["그밖"] = round(100 * cube(P, data, Q2, "L", "그 밖").mean(), 1); d["정답"] = round(100 * cube(P, data, Q2, "L", "정답").mean(), 1); d["첫단어"] = round(100 * np.nanmean(first_word_cube(P, data, Q2, "L")), 1)
        d["기울기x8"] = boot(slope8(Y)); dd = item(Y) - item(Yb[ix]); d["minus_base"] = boot(dd); d["판정"] = eq_rule(d["minus_base"])
        G = item(Yq[ix]) - item(Y); d["G"] = boot(G); d["G_대비_base"] = round(100 * G.mean() / Gb[ix].mean(), 1); d["이점유지"] = bool(d["G"][1] > 0 and d["G_대비_base"] >= 50)
        S[a] = d; print(a, json.dumps(d, ensure_ascii=False))
    if "l2r_m64" in arms and "l2r_m8" in arms:
        P8, P64 = load_pred("dream", "layout", "l2r_m8"), load_pred("dream", "layout", "l2r_m64"); Q3 = [q for q in Q if all((q, c) in P8 and (q, c) in P64 for c in data[q]["ctx"] if c[0] in "LS")]
        S["m64_minus_m8"] = dict(다른값=boot(item(cube(P64, data, Q3, "L")) - item(cube(P8, data, Q3, "L"))), S=boot(np.array([np.mean([L(P64, data, q, c) == "정답" for c in ("S_e", "S_l")]) - np.mean([L(P8, data, q, c) == "정답" for c in ("S_e", "S_l")]) for q in Q3]))); print("m64 - m8", S["m64_minus_m8"])
    json.dump(S, open(f"{RES}/summary_gen_layout.json", "w"), ensure_ascii=False, indent=1)


def natural():
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_anchor.jsonl")}
    def load(n): return {(r["qid"], r["cond"]): r["pred"] for p in glob.glob(f"{RES}/{n}.jsonl") + glob.glob(f"{RES}/{n}.sh*.jsonl") for r in map(json.loads, open(p))}
    Pq, Pb = load("qwen_anchorbs1"), load("dream_anchorbs1_s16"); arms = [a for a in ("l2r_m16", "ent_s4", "ent_s1", "l2r_m64") if glob.glob(f"{RES}/dream_anchor_{a}*.jsonl")]; S = dict(arms=arms)
    F = lambda P, Q, c: np.array([max(f1(P[(q, c)], x) for x in data[q]["answers"]) for q in Q])
    for a in arms:
        P = load(f"dream_anchor_{a}"); Q = [q for q in data if all((q, c) in P for c in ("gold", "hard3"))]
        D = F(P, Q, "gold") - F(P, Q, "hard3"); Db = F(Pb, Q, "gold") - F(Pb, Q, "hard3"); Dq = F(Pq, Q, "gold") - F(Pq, Q, "hard3"); d = dict(n=len(Q))
        d["F1"] = dict(gold=round(100 * F(P, Q, "gold").mean(), 1), hard3=round(100 * F(P, Q, "hard3").mean(), 1)); d["피해"] = boot(D); d["피해_base"] = boot(Db); d["minus_base"] = boot(D - Db); d["판정"] = eq_rule(d["minus_base"])
        G = Dq - D; d["G"] = boot(G); d["G_대비_base"] = round(100 * G.mean() / (Dq - Db).mean(), 1); d["이점유지"] = bool(d["G"][1] > 0 and d["G_대비_base"] >= 50); S[a] = d; print(a, json.dumps(d, ensure_ascii=False))
    json.dump(S, open(f"{RES}/summary_gen_natural.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    natural() if "--natural" in sys.argv else layout()
