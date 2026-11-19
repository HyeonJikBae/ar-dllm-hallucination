"""E29 (사후 점검, CPU): 실제 모집단 (0/5 80%, 1~4/5 11%, 5/5 9%) 에서 궤적 감지가 무엇을 고르는가.  E12 의 문턱 (1~4/5 + 5/5 행의 상위 40%) 을 0/5 행에 그대로 적용."""
import json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from common import k, mid, OUT, MODEL, REV, extract_answer, e7_rows
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); done = np.load(f"{D7}/done.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]; n = len(RR)
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J])
kk = k[RR]; corr = np.array([meta[int(i)]["new_type"] == "correct" for i in RR]); isz, isf, ism = kk == 0, kk == 5, (kk >= 1) & (kk <= 4)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SPs = set(tok.all_special_ids)
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
def dc(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SPs), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
ANS = [[dc(r) for r in np.where(MK[:, s], IDS[:, s], committed)] for s in range(32)]
SC = np.array([np.mean([ANS[s][j] != ANS[31][j] for s in range(31)]) for j in range(n)])
inR = ism | isf; out = {}
W = {"0/5": 44110, "1~4/5 오답": 5950 * 3117 / 5950, "1~4/5 정답": 2833, "5/5": 5165}
for q in (.3, .4, .5):
    th = np.quantile(SC[inR], 1 - q); fl = {"0/5": float((SC[isz] >= th).mean()), "1~4/5 오답": float((SC[ism & ~corr] >= th).mean()), "1~4/5 정답": float((SC[ism & corr] >= th).mean()), "5/5": float((SC[isf] >= th).mean())}
    cnt = {g: fl[g] * W[g] for g in W}; tot = sum(W.values()); flagged = sum(cnt.values())
    out[f"상위{int(q * 100)}% (1~4/5+5/5 기준 문턱)"] = {"집단별 걸린 비율 %": {g: round(100 * v, 1) for g, v in fl.items()}, "모집단에서 다시 생성하는 비율 %": round(100 * flagged / tot, 1),
        "걸린 것 중 1~4/5 오답 몫 %": round(100 * cnt["1~4/5 오답"] / flagged, 1), "걸린 것 중 0/5 몫 %": round(100 * cnt["0/5"] / flagged, 1), "1~4/5 오답 재현율 %": round(100 * fl["1~4/5 오답"], 1)}
    print(f"q={q}", json.dumps(out[f"상위{int(q * 100)}% (1~4/5+5/5 기준 문턱)"], ensure_ascii=False), flush=True)
out["SC 평균"] = {nm: round(float(SC[m].mean()), 3) for nm, m in (("0/5", isz), ("1~4/5 오답", ism & ~corr), ("1~4/5 정답", ism & corr), ("5/5", isf))}
print(out["SC 평균"]); json.dump(out, open(f"{OUT}/e29_population.json", "w"), ensure_ascii=False, indent=1)
