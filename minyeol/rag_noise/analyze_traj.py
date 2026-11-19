"""Dream 생성 궤적 기술 통계 (PREREG_poison.md v5).  usage: python analyze_traj.py     출력: results/summary_traj_bindpatch.json"""
import json, os, re, string
from collections import Counter
import numpy as np
from grade_bindpatch import load, cats

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"; VALS = ["aX", "aY", "aZ", "bX", "bY", "bZ"]


def norm(s):
    s = "".join(c for c in s.lower() if c not in set(string.punctuation)); return re.sub(r"\b(a|an|the)\b", " ", s).split()


def f1(p, g):
    p, g = norm(p), norm(g); c = sum((Counter(p) & Counter(g)).values())
    return 0.0 if c == 0 else 2 * c / (len(p) + len(g))


def main():
    rows = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_bindpatch.jsonl")}
    T = [json.loads(l) for l in open(f"{RES}/traj_dream_bindpatch.jsonl")]
    P = {m: load(m, "bindpatch") for m in ("qwen", "dream")}
    common = {q for q in P["qwen"] if P["qwen"][q]["ok"] and P["dream"][q]["ok"]}
    strict = {q for q in common if all(cats(P[m][q], "link")[0, 1] == 1 for m in P)}
    recs = []
    for t in T:
        ans = "aX" if t["side"] == "src" else "bY"; ia = VALS.index(ans); v = rows[t["qid"]]["v"]
        commit = sum(s["masked"][0] == "1" for s in t["steps"]) - 1
        f = {k: f1(t["pred"], v[k]) for k in VALS}; best = max(f, key=f.get)
        margins = [s["cand"][ia] - max(c for i, c in enumerate(s["cand"]) if i != ia) for s in t["steps"][: commit + 1]]
        recs.append(dict(qid=t["qid"], side=t["side"], commit=commit, step0=t["steps"][0]["argmax"], final0=t["final"][0],
                         ok_tok=t["final"][0] == t["cand"][ia], ok_f1=f[best] >= .5 and best == ans, other_f1=f[best] >= .5 and best != ans,
                         step0_ok=t["steps"][0]["argmax"] == t["cand"][ia], flip=len({s["argmax"] for s in t["steps"][: commit + 1]}) > 1,
                         m0=margins[0], mc=margins[-1]))
    S = {}
    for name, qs in (("전체", set(rows)), ("공통", common), ("strict", strict)):
        R = [r for r in recs if r["qid"] in qs]; c = np.array([r["commit"] for r in R])
        S[name] = dict(n=len(R), commit_dist={str(k): int((c == k).sum()) for k in range(16)}, commit_median=float(np.median(c)),
                       commit_0=round(float(np.mean(c == 0)) * 100, 1), commit_1_3=round(float(np.mean((c >= 1) & (c <= 3))) * 100, 1),
                       commit_4_7=round(float(np.mean((c >= 4) & (c <= 7))) * 100, 1), commit_8p=round(float(np.mean(c >= 8)) * 100, 1),
                       step0_eq_final=round(float(np.mean([r["step0"] == r["final0"] for r in R])) * 100, 1),
                       acc_tok=round(float(np.mean([r["ok_tok"] for r in R])) * 100, 1), acc_f1=round(float(np.mean([r["ok_f1"] for r in R])) * 100, 1),
                       other_f1=round(float(np.mean([r["other_f1"] for r in R])) * 100, 1),
                       step0_ok=round(float(np.mean([r["step0_ok"] for r in R])) * 100, 1),
                       flip_before_commit=round(float(np.mean([r["flip"] for r in R])) * 100, 1),
                       margin_step0=round(float(np.mean([r["m0"] for r in R])), 2), margin_at_commit=round(float(np.mean([r["mc"] for r in R])), 2))
        print(name, json.dumps(S[name], ensure_ascii=False))
    json.dump(dict(summary=S, per_item=recs), open(f"{RES}/summary_traj_bindpatch.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
