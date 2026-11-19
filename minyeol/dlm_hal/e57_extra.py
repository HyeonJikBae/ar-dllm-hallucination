"""E57 (d) 보조 (CPU, 새 GPU 없음): 심사자가 낼 대안 설명마다 이미 있는 자료로 바로 셀 수 있는 숫자.
E49 행 (E44 라벨) 경로별: 채점 근접 오답 (금·별칭이 답 안에 있거나 답이 금 안에 있음), 답 첫 토큰이 Qwen 답 첫 토큰과 같음,
'후보가 있었다가 잃음' 이 DLM 고유 실패 중 차지하는 몫 (주장이 덮는 범위), Dream 최종 칸 0 이 공백·줄바꿈·특수 토큰인 비율."""
import collections, json, re, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from e57_common import *

T = tok(); M = np.where(mid)[0]; WR = ("entity", "broken")
R49 = [int(r) for r in M if E44[int(r)]["new_type"] in WR]
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()
def near(r):
    a = norm(E44[r]["answer"])
    if not a: return False
    gs = [norm(rows[r]["gold_value"])] + [norm(x) for x in rows[r]["gold_aliases"]]
    return any(g and (re.search(r"\b" + re.escape(g) + r"\b", a) or re.search(r"\b" + re.escape(a) + r"\b", g)) for g in gs)
pct = lambda a: round(100 * float(np.mean(a)), 1) if len(a) else None
O = {}
for nm, S_ in (("DLM 고유", [r for r in R49 if q_corr[r]]), ("공유", [r for r in R49 if not q_corr[r]])):
    fq = []
    for r in S_:
        qa, da = QF[r]["model_answer"], E44[r]["answer"]
        if qa and da: fq.append(T(" " + qa)["input_ids"][0] == T(" " + da)["input_ids"][0])
    ft = [E44[r]["final_tok"] for r in S_]
    O[nm] = {"n": len(S_), "채점 근접 오답 (금/별칭 ⊂ 답 또는 답 ⊂ 금) %": pct([near(r) for r in S_]),
             "Dream 답 첫 토큰 = Qwen 답 첫 토큰 %": pct(fq),
             "step0 금 1 위 %": pct([E44[r]["rank0"] == 0 for r in S_]), "step0 금 top-5 %": pct([E44[r]["rank0"] < 5 for r in S_]),
             "1 위였다가 첫 칸에서 잃음 % (주장이 직접 덮는 몫)": pct([E44[r]["lost"] for r in S_]),
             "최종 칸 0 이 공백·줄바꿈·특수 토큰 %": pct([(t in T.all_special_ids) or (T.decode([t]).strip() == "") for t in ft]),
             "붕괴 (broken) %": pct([E44[r]["new_type"] == "broken" for r in S_]),
             "k 평균 (원래)": round(float(np.mean(k[S_])), 2)}
# 근접 오답 (채점 경계) 을 빼도 E49 / E49c 차이가 남는가
R49 = np.array(R49); nr = np.array([near(int(r)) for r in R49]); d = q_corr[R49]; y = np.array([E44[int(r)]["rank0"] == 0 for r in R49])
keep = ~nr
O["근접 오답 뺀 E49 step0 금 1 위 차이 (DLM 고유 − 공유) [사실 군집 CI]"] = boot_facts(fid[R49[keep]], lambda ix: y[keep][ix][d[keep][ix]].mean() - y[keep][ix][~d[keep][ix]].mean())
E27 = {}
for l in open(f"{OUT}/results_e27b.jsonl", encoding="utf-8"):
    x = json.loads(l)
    if x["set"] in ("broken", "entity", "correct"): E27.setdefault(x["row"], {})[x["cond"]] = x["new_type"]
Wc = np.array(sorted(r for r, v in E27.items() if v["base"] in WR)); nc = np.array([near(int(r)) for r in Wc]); dc = q_corr[Wc]; yc = np.array([E27[int(r)]["칸0먼저"] == "correct" for r in Wc])
kc = ~nc
O["근접 오답 뺀 E49c 칸0먼저 복구 차이 (DLM 고유 − 공유) [CI]"] = boot_facts(fid[Wc[kc]], lambda ix: yc[kc][ix][dc[kc][ix]].mean() - yc[kc][ix][~dc[kc][ix]].mean())
O["E49c 행 중 근접 오답 % (DLM 고유 / 공유)"] = [pct(nc[dc]), pct(nc[~dc])]
upd_summary("d_extra", O); print(json.dumps(O, ensure_ascii=False, indent=1, default=str)); print("DONE")
