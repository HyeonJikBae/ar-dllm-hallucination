"""padding 재검증: 기준 세트 gold / hard3 / rand3 를 배치 8 (기존) 과 배치 1 (run_anchor_bs1.sh) 로 생성한 결과 비교 (PREREG_poison.md v2 7번).

보고 (배치마다): F1, 노이즈 피해 (F1 gold - hard3 / rand3, 95% bootstrap) 와 Dream - Qwen 차이 (짝지은 bootstrap),
  EM 공통 집합 (두 모델 모두 gold EM 정답) 크기와 그 위의 hard3 오답 유형 (F1 < 0.5), 실패 차이 중 '방해 문서 개체' 몫,
  배치 8 과 배치 1 의 답 일치율.
출력: 화면 표, results/summary_bs1.json
"""
import collections, json, os
import numpy as np
from grade import norm, boot
from grade_anchor import etype, score

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"
M, C = ["qwen", "dream"], ["gold", "hard3", "rand3"]
FILES = {"bs8": {"qwen": "qwen_anchor", "dream": "dream_anchor_s16"}, "bs1": {"qwen": "qwen_anchorbs1", "dream": "dream_anchorbs1_s16"}}
T = ["방해 문서 개체", "근거 문서 속 다른 개체", "중간 단계 개체", "부분 일치", "문맥에 없음"]


def main():
    data = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{HERE}/data/musique_anchor.jsonl")}
    R = {b: {m: {(r["qid"], r["cond"]): r for r in map(json.loads, open(f"{RES}/{f}.jsonl")) if r["cond"] in C} for m, f in fs.items()} for b, fs in FILES.items()}
    Q = sorted(q for q in data if all((q, c) in R[b][m] for b in R for m in M for c in C))
    em = lambda r, q: any(norm(r["pred"]) == norm(a) for a in data[q]["answers"])
    out = {"n": len(Q)}
    for b in R:
        o = {}
        print(f"\n===== {b}: {len(Q)} 질의 =====")
        for m in M: o[f"F1_{m}"] = {c: round(100 * np.mean([score(R[b][m][(q, c)]) for q in Q]), 1) for c in C}; print(f"  {m:5s} F1", o[f"F1_{m}"])
        for c in ("hard3", "rand3"):
            d = {m: np.array([score(R[b][m][(q, 'gold')]) - score(R[b][m][(q, c)]) for q in Q]) for m in M}
            o[f"drop_{c}"] = {m: boot(d[m]) for m in M}; o[f"drop_{c}"]["dream_minus_qwen"] = boot(d["dream"] - d["qwen"])
            print(f"  피해 {c}: Qwen {o[f'drop_{c}']['qwen']}  Dream {o[f'drop_{c}']['dream']}  Dream-Qwen {o[f'drop_{c}']['dream_minus_qwen']}")
        common = [q for q in Q if all(em(R[b][m][(q, "gold")], q) for m in M)]
        br = {m: [q for q in common if score(R[b][m][(q, "hard3")]) < .5] for m in M}
        ty = {m: collections.Counter(etype(data[q], R[b][m][(q, "hard3")], "hard3") for q in br[m]) for m in M}
        gap, gap_d = len(br["qwen"]) - len(br["dream"]), ty["qwen"]["방해 문서 개체"] - ty["dream"]["방해 문서 개체"]
        o["common"] = dict(n=len(common), broken={m: len(br[m]) for m in M}, types={m: dict(ty[m]) for m in M}, gap=gap, gap_distractor=gap_d)
        print(f"  EM 공통 {len(common)}: hard3 오답 Qwen {len(br['qwen'])} / Dream {len(br['dream'])},  차이 {gap} 중 방해 문서 개체 {gap_d}")
        for m in M: print(f"    {m:5s} " + ", ".join(f"{t} {ty[m][t]}" for t in T))
        out[b] = o
    out["agree"] = {m: {c: round(100 * np.mean([R["bs8"][m][(q, c)]["pred"] == R["bs1"][m][(q, c)]["pred"] for q in Q]), 1) for c in C} for m in M}
    print("\n배치 8 과 배치 1 의 답 일치율 (%):", out["agree"])
    json.dump(out, open(f"{RES}/summary_bs1.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
