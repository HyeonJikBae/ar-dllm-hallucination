"""results/{qwen,dream}.jsonl 채점: 근거가 다 있어도 방해 문서가 생성을 망치는가, AR 과 dLLM 이 다른가.

  1. 모델 x 조건 EM / F1 (SQuAD 정규화, 정답 + 별칭 중 최대)
  2. 노이즈 피해 = 질의별 F1(gold) - F1(top5)  (top5_shuf 도) 의 평균, 95% bootstrap CI
     모델 차이 = Dream 피해 - Qwen 피해 (같은 질의 짝지은 bootstrap)
  3. 깨짐 / 고쳐짐: gold 에서 맞음(F1 >= 0.5) -> top5 에서 틀림, 그 반대
  4. 깨진 답의 출처: 예측이 방해 문서에는 있고 근거 문서에는 없으면 '방해 문서에서 끌려옴'
  5. 근거 수 (2 / 3+) 별
출력: results/summary.json 과 화면 표
"""
import json, os, re, string, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"
MODELS, CONDS = ["qwen", "dream"], ["closed", "gold", "top5", "top5_shuf"]


def norm(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())


def f1(p, g):
    p, g = norm(p).split(), norm(g).split(); c = sum((collections.Counter(p) & collections.Counter(g)).values())
    return 0.0 if c == 0 else 2 * c / (len(p) + len(g))


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float); m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [round(100 * x.mean(), 1), round(100 * np.percentile(m, 2.5), 1), round(100 * np.percentile(m, 97.5), 1)]


def main():
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_full5.jsonl")}
    S = {}                                                           # S[model][cond][qid] = (em, f1, pred)
    for m in MODELS:
        S[m] = {c: {} for c in CONDS}
        for l in open(f"{RES}/{m}.jsonl"):
            r = json.loads(l)
            S[m][r["cond"]][r["qid"]] = (max(float(norm(r["pred"]) == norm(a)) for a in r["answers"]),
                                         max(f1(r["pred"], a) for a in r["answers"]), r["pred"])
    Q = sorted(set.intersection(*[set(S[m][c]) for m in MODELS for c in CONDS]))
    out = {"n_queries": len(Q)}
    print(f"질의 {len(Q)}개 (LACE top-5 에 근거 전부)\n")
    print(f"{'':6s}" + "".join(f"{c:>18s}" for c in CONDS))
    for m in MODELS:
        out[m] = {c: dict(EM=boot([S[m][c][q][0] for q in Q]), F1=boot([S[m][c][q][1] for q in Q])) for c in CONDS}
        print(f"{m:6s}" + "".join(f"   EM {out[m][c]['EM'][0]:5.1f} F1 {out[m][c]['F1'][0]:5.1f}" for c in CONDS))

    print("\n노이즈 피해 = F1(근거만) - F1(top-5)  [평균, 95% CI]")
    for tgt in ("top5", "top5_shuf"):
        d = {m: np.array([S[m]["gold"][q][1] - S[m][tgt][q][1] for q in Q]) for m in MODELS}
        out[f"drop_{tgt}"] = {m: boot(d[m]) for m in MODELS}; out[f"drop_{tgt}"]["dream_minus_qwen"] = boot(d["dream"] - d["qwen"])
        print(f"  {tgt:9s}  Qwen {out[f'drop_{tgt}']['qwen']}   Dream {out[f'drop_{tgt}']['dream']}   Dream-Qwen {out[f'drop_{tgt}']['dream_minus_qwen']}")

    print("\n근거만에서 맞던 답(F1>=0.5)이 top-5 에서 틀린 비율 / 반대로 고쳐진 비율, 깨진 답의 출처")
    for m in MODELS:
        ok = [q for q in Q if S[m]["gold"][q][1] >= 0.5]; bad = [q for q in Q if S[m]["gold"][q][1] < 0.5]
        broke = [q for q in ok if S[m]["top5"][q][1] < 0.5]; fixed = [q for q in bad if S[m]["top5"][q][1] >= 0.5]
        src = collections.Counter()
        for q in broke:
            p = norm(S[m]["top5"][q][2]); r = data[q]
            txt = lambda ids: " ".join(norm(r["docs"][str(i)]["title"] + " " + r["docs"][str(i)]["text"]) for i in ids)
            if not p: src["빈 답"] += 1
            elif p in txt(r["distractors"]) and p not in txt(r["gold"]): src["방해 문서에서 끌려옴"] += 1
            elif p in txt(r["gold"]): src["근거 문서의 다른 개체"] += 1
            else: src["문서에 없음 (환각)"] += 1
        out[f"flip_{m}"] = dict(gold_correct=len(ok), broken=len(broke), broken_rate=round(100 * len(broke) / max(len(ok), 1), 1),
                                gold_wrong=len(bad), fixed=len(fixed), broken_source=dict(src))
        print(f"  {m:6s} 깨짐 {len(broke)}/{len(ok)} ({out[f'flip_{m}']['broken_rate']}%)  고쳐짐 {len(fixed)}/{len(bad)}   출처 {dict(src)}")

    print("\n근거 수별 F1 (근거만 -> top-5)")
    for k, sel in (("2", lambda r: len(r["gold"]) == 2), ("3+", lambda r: len(r["gold"]) >= 3)):
        qs = [q for q in Q if sel(data[q])]; out[f"by_ngold_{k}"] = {}
        for m in MODELS:
            g, t = np.mean([S[m]["gold"][q][1] for q in qs]) * 100, np.mean([S[m]["top5"][q][1] for q in qs]) * 100
            out[f"by_ngold_{k}"][m] = dict(n=len(qs), gold=round(g, 1), top5=round(t, 1))
            print(f"  근거 {k:2s} (n={len(qs)})  {m:6s} {g:5.1f} -> {t:5.1f}  (방해 문서 {5 - int(k[0])}{'개 이하' if k == '3+' else '개'})")
    json.dump(out, open(f"{RES}/summary.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
