"""patch_question.py 채점 (PREREG_poison.md v12).  usage: python grade_question.py     출력: results/summary_question.json"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"; B = 10000
LAYERS = ["emb"] + [str(l) for l in range(0, 28, 2)] + ["27"]; BANDS = {"초반": range(1, 6), "중반": range(6, 11), "후반": range(11, 16)}
SITES = ["fact", "link", "ctx", "qall", "last", "ctrl_f", "ctrl_c"]


def rates(rec, fm, site, k=0):
    """[16] 층별 비율, 두 방향 평균.  k = 0 반영, 1 유지, 2 Z값."""
    out = np.zeros(len(LAYERS))
    for d, (donor, recv) in (("QA>QB", (0, 1)), ("QB>QA", (1, 0))):
        out += 0.5 * (np.array(rec["patch"][fm][d][site]) == rec["cand"][[donor, recv, 2][k]])
    return out


def boot(x, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [round(100 * float(x.mean()), 1), round(100 * float(np.percentile(bs, 2.5)), 1), round(100 * float(np.percentile(bs, 97.5)), 1), round(float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))), 4)]


def holm(ps):
    o = np.argsort(ps); adj = [0.0] * len(ps); run = 0.0
    for i, j in enumerate(o): run = max(run, (len(ps) - i) * ps[j]); adj[j] = round(float(min(1, run)), 4)
    return adj


def main():
    Q = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{RES}/qpair_qwen.jsonl")}
    D = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{RES}/qpair_dream.jsonl")}
    S = dict(n=len(Q), qwen_base_ok=sum(r["ok"]["base"] for r in Q.values()), qwen_sand_ok=sum(r["ok"]["sand"] for r in Q.values()),
             qwen_both_ok=sum(r["ok"]["base"] and r["ok"]["sand"] for r in Q.values()), dream_tested=len(D), dream_ok=sum(r["ok"]["base"] for r in D.values()))
    main_q = [q for q, r in D.items() if r["ok"]["base"] and Q[q]["ok"]["base"] and Q[q]["ok"]["sand"]]; S["n_main"] = len(main_q)
    print(json.dumps(S, ensure_ascii=False))
    san = [r for r in Q.values() if "base" in r["patch"]]; bad = 0
    for r in san:
        for site in ("fact", "link", "ctx", "ctrl_f", "ctrl_c"):
            bad += int(rates(r, "base", site, 0).sum() > 0); bad += int(max(r["dist"]["base"][site]) > 0)
    S["H12_0"] = dict(n_pairs=len(san), 위반=bad); print(f"H12.0 구조 확인: {len(san)} 쌍, 위반 {bad}")
    groups = {"qwen_sand": ([Q[q] for q in main_q], "sand"), "dream": ([D[q] for q in main_q], "base")}
    arr = {g: {s: np.stack([rates(r, fm, s) for r in rs]) for s in SITES} for g, (rs, fm) in groups.items()}
    S["curves"] = {g: {s: (a.mean(0) * 100).round(1).tolist() for s, a in v.items()} for g, v in arr.items()}
    S["dist"] = {g: {s: np.mean([r["dist"][fm][s] for r in rs], 0).round(4).tolist() for s in SITES} for g, (rs, fm) in groups.items()}
    S["유지"] = {g: {s: (np.stack([rates(r, fm, s, 1) for r in rs]).mean(0) * 100).round(1).tolist() for s in ("fact", "ctx")} for g, (rs, fm) in groups.items()}
    S["qfront"] = (np.stack([rates(r, "sand", "qfront") for r in groups["qwen_sand"][0]]).mean(0) * 100).round(1).tolist()
    print("\n반영 % (층: " + " ".join(LAYERS) + ")")
    for g in arr:
        for s in SITES: print(f"  {g:10s} {s:7s} " + " ".join(f"{v:5.1f}" for v in S["curves"][g][s]))
    print(f"  qwen_sand  qfront  " + " ".join(f"{v:5.1f}" for v in S["qfront"]))
    print("\nhidden state 차이 (fact / ctrl_f)")
    for g in arr: print(f"  {g:10s} fact   " + " ".join(f"{v:5.2f}" for v in S["dist"][g]["fact"])); print(f"  {g:10s} ctrl_f " + " ".join(f"{v:5.2f}" for v in S["dist"][g]["ctrl_f"]))
    band = lambda a, b: a[:, list(b)].mean(1)
    for h, g in (("H12.1", "dream"), ("H12.2", "qwen_sand")):
        d = {k: boot(band(arr[g]["fact"], b) - band(arr[g]["ctrl_f"], b)) for k, b in BANDS.items()}
        ad = holm([d[k][3] for k in BANDS])
        S[h] = {k: dict(diff=d[k][:3], p_holm=ad[i]) for i, k in enumerate(BANDS)}; print(h, g, S[h])
    S["H12.3"] = {k: boot((band(arr["dream"]["fact"], b) - band(arr["dream"]["ctrl_f"], b)) - (band(arr["qwen_sand"]["fact"], b) - band(arr["qwen_sand"]["ctrl_f"], b)))[:3] for k, b in BANDS.items()}
    print("H12.3 Dream - qwen_sand:", S["H12.3"])
    json.dump(S, open(f"{RES}/summary_question.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
