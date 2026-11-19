"""E57 (c) (CPU): 라벨 신뢰도.  Dream 원래 라벨 (rows) 대 다시 생성 E7 / E44 (다른 묶음 구성) 의 일치를 경로별로, 경로 라벨이 바뀌는 비율.
경로 = Dream 1~4/5 오답 (entity/broken) 중 같은 문구 Qwen 정오 (Qwen 라벨은 한 번 실행으로 고정이므로 경로가 바뀌는 길은 Dream 오답 여부가 바뀌는 것뿐).
추가: Qwen 라벨 재현 (e57_qwen_gold.py 의 첫 자리 1 위 토큰이 Qwen 답의 첫 토큰과 같은가), E49 결과 (step0 금 1 위) 의 E7 대 E44 재현."""
import collections, json, sys
import numpy as np
sys.path.insert(0, "followup_mk")
from e57_common import *

M = np.where(mid)[0]; WR = ("entity", "broken")
lab = {"원래": {int(r): ty[r] for r in M}, "E7": {int(r): E7T[int(r)] for r in M}, "E44": {int(r): E44[int(r)]["new_type"] for r in M}}
O = {"행 (1~4/5)": int(len(M))}
pct = lambda a: round(100 * float(np.mean(a)), 1) if len(a) else None
pair = {}
for a, b in (("원래", "E7"), ("원래", "E44"), ("E7", "E44")):
    A, Bm = lab[a], lab[b]
    ca = np.array([A[r] == "correct" for r in M]); cb = np.array([Bm[r] == "correct" for r in M])
    pair[f"{a} 대 {b}"] = {"정오 일치 %": pct(ca == cb), "4 유형 일치 %": pct([A[r] == Bm[r] for r in M]),
                           "원래 쪽 오답 (entity/broken) 중 다른 쪽 정답 %": pct([Bm[r] == "correct" for r in M if A[r] in WR]),
                           "원래 쪽 정답 중 다른 쪽 오답 %": pct([Bm[r] != "correct" for r in M if A[r] == "correct"])}
O["실행 쌍 일치"] = pair

def route(L, r): return None if L[r] not in WR else ("DLM 고유" if q_corr[r] else "공유")
BYR = {}
for a, b in (("원래", "E7"), ("원래", "E44"), ("E7", "E44")):
    A, Bm = lab[a], lab[b]; o = {}
    for rt in ("DLM 고유", "공유"):
        S_ = [r for r in M if route(A, r) == rt]
        o[rt] = {"행 (기준 실행)": len(S_), "다른 실행에서 정답 (경로 라벨 사라짐) %": pct([Bm[r] == "correct" for r in S_]), "다른 실행에서 abstain %": pct([Bm[r] == "abstain" for r in S_]),
                 "entity<->broken 바뀜 %": pct([Bm[r] in WR and Bm[r] != A[r] for r in S_])}
    # Dream 정답 행이 다른 실행에서 오답이 되어 새 경로 라벨을 얻는 비율 (Qwen 정오별)
    for qc, nm in ((True, "Qwen 맞힘 (생기면 DLM 고유)"), (False, "Qwen 틀림 (생기면 공유)")):
        S_ = [r for r in M if A[r] == "correct" and q_corr[r] == qc]
        o[f"기준 실행 정답 중 다른 실행 오답: {nm} %"] = [pct([Bm[r] in WR for r in S_]), len(S_)]
    U = [r for r in M if route(A, r) or route(Bm, r)]; ch = [route(A, r) != route(Bm, r) for r in U]
    o["어느 쪽이든 경로 라벨이 있는 행 중 라벨이 다른 비율 %"] = [pct(ch), len(U)]
    for rt in ("DLM 고유", "공유"):
        U2 = [r for r in M if route(A, r) == rt or route(Bm, r) == rt]; o[f"{rt}: 어느 쪽이든 이 경로인 행 중 한쪽에만 있음 %"] = [pct([route(A, r) != route(Bm, r) for r in U2]), len(U2)]
    BYR[f"{a} -> {b}"] = o
O["경로별 라벨 안정성"] = BYR
# 세 실행 모두 오답인 행 (튼튼한 핵) 에서 E49
core = [r for r in M if all(lab[x][r] in WR for x in lab)]
y = lambda rs: pct([E44[r]["rank0"] == 0 for r in rs])
O["세 실행 모두 오답인 행에서 E49 step0 금 1 위 % (DLM 고유 / 공유, n)"] = [y([r for r in core if q_corr[r]]), y([r for r in core if not q_corr[r]]), sum(q_corr[r] for r in core), sum(not q_corr[r] for r in core)]
O["원래 라벨로 정한 경로의 E49 % (DLM 고유 / 공유, n)"] = [y([r for r in M if route(lab['원래'], r) == "DLM 고유"]), y([r for r in M if route(lab['원래'], r) == "공유"]),
                                                       sum(route(lab['원래'], r) == "DLM 고유" for r in M), sum(route(lab['원래'], r) == "공유" for r in M)]
# E49 결과 재현: E7 step0 칸0 argmax == 금 첫 토큰 (E7 은 다른 묶음 구성)
IDS = np.load(f"{D7}/ids.npy", mmap_mode="r")
R49 = [r for r in M if lab["E44"][r] in WR]
a44 = np.array([E44[r]["argmax0"] for r in R49]); a7 = np.array([int(IDS[W7[r], 0, 0]) for r in R49]); g = np.array([E44[r]["gold_id"] for r in R49]); dq = q_corr[R49]
O["E49 결과 재현 (E44 대 E7 step0 칸0 argmax)"] = {"argmax 일치 %": pct(a44 == a7), "E7 기준 금 1 위 % (DLM 고유 / 공유)": [pct((a7 == g)[dq]), pct((a7 == g)[~dq])],
                                                 "E44 기준 (E49 원값)": [pct((a44 == g)[dq]), pct((a44 == g)[~dq])]}
# S 행: E8 base, E27b base, E7 의 일치
def load(fn):
    R = {}
    for l in open(f"{OUT}/{fn}", encoding="utf-8"):
        x = json.loads(l)
        if x["cond"] == "base" and x["set"] in ("broken", "entity", "correct"): R[x["row"]] = x["new_type"]
    return R
b8, b27 = load("results_e8.jsonl"), load("results_e27b.jsonl"); Sr = sorted(b8)
O["S 행 base 일치 (E8 base / E27b base / E7)"] = {"행": len(Sr), "E8 = E27b 4 유형 %": pct([b8[r] == b27[r] for r in Sr]), "E8 = E7 4 유형 %": pct([b8[r] == E7T[r] for r in Sr]),
                                                  "E8 = 원래 4 유형 %": pct([b8[r] == ty[r] for r in Sr])}
# 다시 생성만으로 생기는 '복구' (잡음 바닥): 원래 오답 -> E7 정답, S 행 경로별
for nm, A, Bm in (("원래 -> E7", lab["원래"], lab["E7"]), ("원래 -> E44", lab["원래"], lab["E44"])):
    O[f"잡음 바닥: S 행 중 {nm} 정답 % (DLM 고유 / 공유, n)"] = [pct([Bm[r] == "correct" for r in Sr if route(A, r) == "DLM 고유"]), pct([Bm[r] == "correct" for r in Sr if route(A, r) == "공유"]),
                                                            sum(route(A, r) == "DLM 고유" for r in Sr), sum(route(A, r) == "공유" for r in Sr)]
# Qwen 라벨 재현
from e57_common import tok
T = tok()
def first_match(r):
    a = QF[r]["model_answer"]
    if not a: return None
    s = T.decode([int(qwen(r, "top1_id"))]).strip()
    return bool(s) and a.startswith(s)
QR = {}
for nm, S_ in (("1~4/5 전체", list(M)), ("DLM 고유 (E44 라벨)", [r for r in R49 if q_corr[r]]), ("공유 (E44 라벨)", [r for r in R49 if not q_corr[r]])):
    fm = [first_match(r) for r in S_]; fm = [x for x in fm if x is not None]
    QR[nm] = {"Qwen forward 1 위 토큰으로 Qwen 답이 시작 %": pct(fm), "Qwen forward 1 위 = 금 첫 토큰 %": pct([qwen(r, "gold_rank") == 0 for r in S_]), "n": len(S_)}
QR["Qwen 맞힘 행 중 Qwen 1 위 = 금 첫 토큰 % (나머지는 별칭·다른 첫 토큰으로 맞힘)"] = pct([qwen(r, "gold_rank") == 0 for r in M if q_corr[r]])
QR["Qwen 틀림 행 중 Qwen 1 위 = 금 첫 토큰 %"] = pct([qwen(r, "gold_rank") == 0 for r in M if not q_corr[r]])
O["Qwen 라벨 재현"] = QR
upd_summary("c_labels", O); print(json.dumps(O, ensure_ascii=False, indent=1, default=str)); print("DONE")
