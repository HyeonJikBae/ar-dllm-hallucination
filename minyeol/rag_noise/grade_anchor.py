"""기준 세트 채점: results/qwen_anchor.jsonl, results/dream_anchor_s{16,8,4,2}.jsonl  (1,000 질의 x closed / gold / hard3 / rand3).
사전 기준은 PREREG.md.  기본 비교 = Qwen vs Dream steps 16 (단계당 1 토큰); --sweep 이면 Dream 병렬도별 표 추가.

  1. 모델 x 조건 EM / F1 (SQuAD 정규화, 정답 + 별칭 중 최대)
  2. 노이즈 피해 = 질의별 F1(gold) - F1(hard3 | rand3), 95% bootstrap CI, Dream - Qwen 차이 (짝지은 bootstrap)
  3. 오답 유형 (F1 < 0.5): 중간 단계 개체 (질문 분해의 중간 답) / 부분 일치 / 근거 문서 속 다른 개체 / 방해 문서 개체 / 문맥에 없음
  4. 깨짐 = gold 정답 -> 노이즈 조건 오답, 그 유형
  5. 민감도: 모든 조건의 프롬프트가 2,048 토큰 이하인 질의만 (Dream 학습 길이 확인용)
  6. 깨진 답의 형태 (PREREG): 읽기형 / 혼합형 / 그 밖,  전제 감사: EM, Acc (정답 포함), F1, 복사율, 답 길이
출력: results/summary_anchor.json 과 화면 표
"""
import argparse, collections, json, os
import numpy as np
from grade import norm, f1, boot

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"
MODELS, CONDS, NOISE = ["qwen", "dream"], ["closed", "gold", "hard3", "rand3"], ["hard3", "rand3"]
TYPES = ["정답", "중간 단계 개체", "부분 일치", "근거 문서 속 다른 개체", "방해 문서 개체", "문맥에 없음"]


def load(dream_steps=16):
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_anchor.jsonl")}
    f = {"qwen": f"{RES}/qwen_anchor.jsonl", "dream": f"{RES}/dream_anchor_s{dream_steps}.jsonl"}
    R = {m: {(r["qid"], r["cond"]): r for r in map(json.loads, open(f[m]))} for m in MODELS}
    return data, R


def ctx_text(d, c):
    ids = d["ctx"][c] or []
    return {i: norm(d["docs"][str(i)]["title"] + " " + d["docs"][str(i)]["text"]) for i in ids}


def form(d, r, c):
    """PREREG 의 깨진 답 형태: 읽기형 / 혼합형 / 그 밖."""
    n = norm(r["pred"]); toks = n.split(); P = ctx_text(d, c)
    if not toks: return "그 밖"
    if any(n in t for t in P.values()): return "읽기형"
    cov = [w for w in toks if any(w in t.split() for t in P.values())]
    src = {i for w in toks for i, t in P.items() if w in t.split()}
    return "혼합형" if len(toks) >= 2 and len(cov) / len(toks) >= .8 and len(src) >= 2 else "그 밖"


def audit(data, R, m, c, Q):
    """전제 감사: EM / Acc (정규화 정답이 예측에 포함) / F1 / 복사율 (예측이 문맥에 연속으로 있음) / 답 길이."""
    em = np.mean([max(float(norm(R[m][(q, c)]["pred"]) == norm(a)) for a in data[q]["answers"]) for q in Q])
    acc = np.mean([max(float(norm(a) in norm(R[m][(q, c)]["pred"])) for a in data[q]["answers"]) for q in Q])
    f = np.mean([score(R[m][(q, c)]) for q in Q])
    cp = np.mean([float(bool(norm(R[m][(q, c)]["pred"])) and any(norm(R[m][(q, c)]["pred"]) in t for t in ctx_text(data[q], c).values())) for q in Q]) if c != "closed" else float("nan")
    ln = np.mean([len(R[m][(q, c)]["pred"].split()) for q in Q])
    return dict(EM=round(100 * em, 1), Acc=round(100 * acc, 1), F1=round(100 * f, 1), copy=round(100 * cp, 1), len=round(ln, 2))


def score(r): return max(f1(r["pred"], a) for a in r["answers"])


def etype(d, r, c):
    p = r["pred"]; f = score(r)
    if f >= .5: return "정답"
    if any(f1(p, b) >= .5 for b in d["bridges"]): return "중간 단계 개체"
    if f > 0: return "부분 일치"
    n = norm(p); txt = lambda ids: " ".join(norm(d["docs"][str(i)]["title"] + " " + d["docs"][str(i)]["text"]) for i in ids)
    if n and n in txt(d["gold"]): return "근거 문서 속 다른 개체"
    if n and c in NOISE and n in txt([i for i in d["ctx"][c] if i not in d["gold"]]): return "방해 문서 개체"
    return "문맥에 없음"


def report(data, R, Q, tag):
    out = {"n": len(Q)}
    print(f"\n===== {tag}: 질의 {len(Q)}개 =====")
    print(f"{'':7s}" + "".join(f"{c:>17s}" for c in CONDS))
    for m in MODELS:
        out[m] = {c: dict(EM=boot([max(float(norm(R[m][(q, c)]['pred']) == norm(a)) for a in data[q]['answers']) for q in Q]),
                          F1=boot([score(R[m][(q, c)]) for q in Q])) for c in CONDS}
        print(f"{m:7s}" + "".join(f"  EM {out[m][c]['EM'][0]:5.1f} F1 {out[m][c]['F1'][0]:5.1f}" for c in CONDS))
    print("노이즈 피해 F1(gold) - F1(노이즈)  [평균, 95% CI]")
    for c in NOISE:
        d = {m: np.array([score(R[m][(q, 'gold')]) - score(R[m][(q, c)]) for q in Q]) for m in MODELS}
        out[f"drop_{c}"] = {m: boot(d[m]) for m in MODELS}; out[f"drop_{c}"]["dream_minus_qwen"] = boot(d["dream"] - d["qwen"])
        print(f"  {c}:  Qwen {out[f'drop_{c}']['qwen']}  Dream {out[f'drop_{c}']['dream']}  Dream-Qwen {out[f'drop_{c}']['dream_minus_qwen']}")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep", action="store_true"); a = ap.parse_args()
    data, R = load()
    Q = sorted(q for q in data if all((q, c) in R[m] for m in MODELS for c in CONDS))
    out = {"all": report(data, R, Q, "전체")}
    short = [q for q in Q if all(R[m][(q, c)]["n_prompt_tok"] <= 2048 for m in MODELS for c in CONDS)]
    out["le2048"] = report(data, R, short, "프롬프트 2,048 토큰 이하")

    print("\n오답 유형 (전체 질의)")
    print(f"{'':15s}" + "".join(f"{t:>14s}" for t in TYPES))
    out["types"] = {}
    for c in ("gold",) + tuple(NOISE):
        for m in MODELS:
            cnt = collections.Counter(etype(data[q], R[m][(q, c)], c) for q in Q); out["types"][f"{m}_{c}"] = dict(cnt)
            print(f"  {m:5s} {c:6s}  " + "".join(f"{cnt[t]:>14d}" for t in TYPES))
    print("\n깨짐 = gold 정답 -> 노이즈 오답, 유형")
    out["broken"] = {}
    for c in NOISE:
        for m in MODELS:
            ok = [q for q in Q if score(R[m][(q, "gold")]) >= .5]
            br = collections.Counter(etype(data[q], R[m][(q, c)], c) for q in ok if score(R[m][(q, c)]) < .5)
            fx = sum(score(R[m][(q, c)]) >= .5 for q in Q if score(R[m][(q, "gold")]) < .5)
            out["broken"][f"{m}_{c}"] = dict(gold_correct=len(ok), broken=sum(br.values()), fixed=fx, types=dict(br))
            print(f"  {m:5s} {c:6s} 깨짐 {sum(br.values()):3d}/{len(ok)} ({100 * sum(br.values()) / len(ok):4.1f}%)  고쳐짐 {fx:3d}  "
                  + ", ".join(f"{t} {br[t]}" for t in TYPES[1:] if br[t]))
    g_wrong = {m: {q for q in Q if score(R[m][(q, "gold")]) < .5} for m in MODELS}
    out["gold_wrong_overlap"] = dict(both=len(g_wrong["qwen"] & g_wrong["dream"]), qwen_only=len(g_wrong["qwen"] - g_wrong["dream"]),
                                     dream_only=len(g_wrong["dream"] - g_wrong["qwen"]))
    print(f"\n근거만에서 틀린 질의: 둘 다 {out['gold_wrong_overlap']['both']}, Qwen 만 {out['gold_wrong_overlap']['qwen_only']}, Dream 만 {out['gold_wrong_overlap']['dream_only']}")
    print("\n전제 감사 (EM / Acc / F1 / 복사율 / 답 길이)")
    out["audit"] = {f"{m}_{c}": audit(data, R, m, c, Q) for m in MODELS for c in CONDS}
    for k, v in out["audit"].items(): print(f"  {k:12s} {v}")
    print("\n깨진 답 형태 (PREREG)")
    out["forms"] = {}
    for c in NOISE:
        for m in MODELS:
            br = [q for q in Q if score(R[m][(q, "gold")]) >= .5 and score(R[m][(q, c)]) < .5]
            cnt = collections.Counter(form(data[q], R[m][(q, c)], c) for q in br); out["forms"][f"{m}_{c}"] = dict(n=len(br), **cnt)
            print(f"  {m:5s} {c:6s} n={len(br):3d}  " + "  ".join(f"{k} {cnt[k]} ({100 * cnt[k] / max(len(br), 1):.0f}%)" for k in ("읽기형", "혼합형", "그 밖")))
    if a.sweep: out["sweep"] = sweep(data, Q)
    json.dump(out, open(f"{RES}/summary_anchor.json", "w"), ensure_ascii=False, indent=1)


def sweep(data, Q):
    """Dream 병렬도별: F1 (gold / hard3 / rand3), 노이즈 피해, 깨진 사례와 형태, NFE (= steps)."""
    print("\nDream 병렬도 sweep (생성 16 토큰)")
    print(f"  {'steps':>5s} {'토큰/단계':>8s} {'gold':>6s} {'hard3':>6s} {'rand3':>6s} {'피해h':>6s} {'피해r':>6s}  깨짐(hard3)  읽기형  혼합형  그밖")
    res = {}
    for st in (16, 8, 4, 2):
        f = f"{RES}/dream_anchor_s{st}.jsonl"
        if not os.path.exists(f): continue
        Rd = {(r["qid"], r["cond"]): r for r in map(json.loads, open(f))}
        F = {c: np.mean([score(Rd[(q, c)]) for q in Q]) * 100 for c in ("gold", "hard3", "rand3")}
        br = [q for q in Q if score(Rd[(q, "gold")]) >= .5 and score(Rd[(q, "hard3")]) < .5]
        cnt = collections.Counter(form(data[q], Rd[(q, "hard3")], "hard3") for q in br)
        res[st] = dict(nfe=st, F1=F, drop_hard3=F["gold"] - F["hard3"], drop_rand3=F["gold"] - F["rand3"], broken=len(br), forms=dict(cnt))
        print(f"  {st:5d} {16 // st:8d} {F['gold']:6.1f} {F['hard3']:6.1f} {F['rand3']:6.1f} {F['gold'] - F['hard3']:6.1f} {F['gold'] - F['rand3']:6.1f}"
              f"  {len(br):10d}  " + "  ".join(f"{100 * cnt[k] / max(len(br), 1):5.0f}%" for k in ("읽기형", "혼합형", "그 밖")))
    return res


if __name__ == "__main__":
    main()
