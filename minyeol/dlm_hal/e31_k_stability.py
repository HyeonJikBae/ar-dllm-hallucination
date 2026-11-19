"""E31 (PREREG.md, CPU): '1~4/5' 범주의 수치 안정성.  E7 (다른 배치 구성 재생성) 으로 사실별 k' 를 다시 세어 원래 k 와 비교."""
import collections, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import k, fid, mid, OUT, ty
D7 = f"{OUT}/e7"; meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
c7 = {int(r): d["new_type"] == "correct" for r, d in meta.items()}
byf = collections.defaultdict(list)
for i in np.where(mid)[0]: byf[fid[i]].append(int(i))
full = {f: v for f, v in byf.items() if len(v) == 5 and all(i in c7 for i in v)}
K0 = np.array([int(k[v[0]]) for v in full.values()]); K1 = np.array([sum(c7[i] for i in v) for v in full.values()])
O = {"사실 수": len(full), "행 단위 정오 일치 %": round(100 * float(np.mean([c7[i] == (ty[i] == "correct") for v in full.values() for i in v])), 2)}
O["k 별 k' 분포"] = {int(a): {int(b): int(((K0 == a) & (K1 == b)).sum()) for b in range(6)} for a in range(1, 5)}
O["1~4/5 에서 0/5 로 %"] = round(100 * float((K1 == 0).mean()), 1); O["1~4/5 에서 5/5 로 %"] = round(100 * float((K1 == 5).mean()), 1)
O["범주 이탈 합 %"] = round(100 * float(((K1 == 0) | (K1 == 5)).mean()), 1); O["k 그대로 %"] = round(100 * float((K0 == K1).mean()), 1)
O["|k-k'| 평균"] = round(float(np.abs(K0 - K1).mean()), 3)
for nm, m in (("0/5 행", k == 0), ("5/5 행", k == 5)):
    v = [i for i in np.where(m)[0] if int(i) in c7]; flip = float(np.mean([c7[int(i)] != (ty[i] == "correct") for i in v])) if v else float("nan")
    O[f"{nm} 뒤집힘 %"] = round(100 * flip, 2); O[f"{nm} 범주 이탈 확률 (5 문구 중 하나 이상) %"] = round(100 * (1 - (1 - flip) ** 5), 1); O[f"{nm} 표본"] = len(v)
O["판정"] = "상당 부분 잡음" if O["범주 이탈 합 %"] >= 25 else ("안정" if O["범주 이탈 합 %"] <= 10 else "중간")
print(json.dumps(O, ensure_ascii=False, indent=1)); json.dump(O, open(f"{OUT}/e31_summary.json", "w"), ensure_ascii=False, indent=1)
