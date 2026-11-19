"""실험 C (H3 시간 실패): Dream 디노이징 궤적에서 '중간에 맞았다가 최종에 틀림' 을 센다.

입력: results/dream_anchor_s{steps}.jsonl 과 _trace.npz (gen.py --trace; 줄 순서 = trace 행 순서).
단계 i 의 현재 답 = masked[i] 칸은 argmax[i], 나머지 칸은 final (Dream 은 확정 토큰을 바꾸지 않음) -> 첫 줄 -> F1 >= 0.5 이면 맞음.
  ever-pass = 어느 단계에서든 맞음,  final = 최종 생성 (jsonl pred) 이 맞음.
보고 (steps 마다):
  - 조건별 ever-pass / final 정답률
  - PREREG H3: hard3 에서 노이즈로 깨진 사례 (gold 맞음 -> hard3 틀림) 중 ever-pass 비율  vs  gold 조건 오답 중 ever-pass 비율
출력: results/trace_summary.json
"""
import json, os
import numpy as np
from transformers import AutoTokenizer
from gen import first_line
from grade import f1

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"


def main():
    tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True)
    out = {}
    for st in (16, 8, 4, 2):
        fj, fz = f"{RES}/dream_anchor_s{st}.jsonl", f"{RES}/dream_anchor_s{st}_trace.npz"
        if not (os.path.exists(fj) and os.path.exists(fz)): continue
        rows = [json.loads(l) for l in open(fj)]; z = np.load(fz); A, Mk, Fn = z["argmax"], z["masked"], z["final"]
        ok = lambda txt, ans: max(f1(first_line(txt), a) for a in ans) >= .5
        ever, fin = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
        for j, r in enumerate(rows):
            fin[j] = max(f1(r["pred"], a) for a in r["answers"]) >= .5
            cur = np.where(Mk[j], A[j], Fn[j][None, :])                 # [steps, L]
            ever[j] = fin[j] or any(ok(tok.decode(c, skip_special_tokens=True), r["answers"]) for c in cur)
        key = {(r["qid"], r["cond"]): j for j, r in enumerate(rows)}
        Q = sorted({r["qid"] for r in rows})
        res = {"by_cond": {}}
        for c in ("closed", "gold", "hard3", "rand3"):
            js = [key[(q, c)] for q in Q]
            res["by_cond"][c] = dict(final=round(100 * fin[js].mean(), 1), ever=round(100 * ever[js].mean(), 1))
        br = [key[(q, "hard3")] for q in Q if fin[key[(q, "gold")]] and not fin[key[(q, "hard3")]]]
        gw = [key[(q, "gold")] for q in Q if not fin[key[(q, "gold")]]]
        res["H3"] = dict(broken_hard3=len(br), broken_ever_pass=round(100 * ever[br].mean(), 1) if br else None,
                         gold_wrong=len(gw), gold_wrong_ever_pass=round(100 * ever[gw].mean(), 1) if gw else None)
        out[st] = res
        print(f"[steps {st:2d}] " + "  ".join(f"{c} final {v['final']:5.1f} ever {v['ever']:5.1f}" for c, v in res["by_cond"].items()))
        print(f"           H3: 깨진 사례 {len(br)} 중 ever-pass {res['H3']['broken_ever_pass']}%  |  gold 오답 {len(gw)} 중 ever-pass {res['H3']['gold_wrong_ever_pass']}%")
    json.dump(out, open(f"{RES}/trace_summary.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
