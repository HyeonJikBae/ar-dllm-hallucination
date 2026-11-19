"""E11 (사후 탐색, PREREG 밖, CPU): 생성 뒤 궤적 신호 (TRE, Time Is a Feature 류) 로 아는데 틀림 감지 + '감지 뒤 조향해서 다시 생성' 두 번 생성 모의.
첫 생성 = E7 (개입 없음), 다시 생성 = E9/E8 조건 결과 (따로 실행한 생성이라 같은 실행 base 가 필요 없음)."""
import collections, json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import rows, grp, k, fid, NF, OUT, MODEL, REV, extract_answer, set_S
D7 = f"{OUT}/e7"; O = {}
R7 = np.load(f"{D7}/rows.npy"); done = np.load(f"{D7}/done.npy"); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]; n = len(RR)
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); LP = np.asarray(np.load(f"{D7}/lp.npy", mmap_mode="r")[J]).astype(np.float32)
MG = np.asarray(np.load(f"{D7}/mg.npy", mmap_mode="r")[J]).astype(np.float32); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J])
kk = k[RR]; g7 = grp[RR]; corr = np.array([meta[int(i)]["new_type"] == "correct" for i in RR]); fid7 = fid[RR]
isz, isf, ism = kk == 0, kk == 5, (kk >= 1) & (kk <= 4); mw = ism & ~corr
W = np.where(isz, 44110 / isz.sum(), np.where(isf, 5165 / isf.sum(), 1.0))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SP = np.array(sorted(tok.all_special_ids))
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); cstep = lastm + 1                      # 칸마다 확정된 step (마지막 가려진 step + 1)
committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
isSP = np.isin(committed, SP); L = np.where(isSP.any(1), isSP.argmax(1), 32); L = np.maximum(L, 1)          # 답 내용 칸 수
cont = np.arange(32)[None, :] < L[:, None]
lp_c = np.take_along_axis(LP, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]                              # 확정 순간의 top-1 logprob
x0 = lambda s: np.where(MK[:, s], IDS[:, s], committed)
def dc(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in set(SP.tolist())), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
ANS = [[dc(r) for r in x0(s)] for s in range(32)]; fin = ANS[31]
F = {}   # 높을수록 '틀림' 쪽
F["첫칸 확정 step"] = cstep[:, 0].astype(float)
F["답 칸 평균 확정 step"] = (cstep * cont).sum(1) / cont.sum(1)
F["확정 logprob 최소(-)"] = -np.where(cont, lp_c, 0).min(1)
F["확정 logprob 평균(-)"] = -(lp_c * cont).sum(1) / cont.sum(1)
F["TRE식 (늦게 확정 x 불확실)"] = ((-lp_c) * (cstep / 32.0) * cont).sum(1) / cont.sum(1)
chg = (IDS[:, 1:] != IDS[:, :-1]) & MK[:, 1:] & MK[:, :-1]
F["가려진 동안 흔들림"] = (chg & cont[:, None, :]).sum((1, 2)).astype(float)
F["중간답 종류"] = np.array([len({ANS[s][j] for s in range(32)}) for j in range(n)], float)
F["최종답과 다른 step 비율"] = np.array([np.mean([ANS[s][j] != fin[j] for s in range(31)]) for j in range(n)])
F["step0 답 != 최종답"] = np.array([ANS[0][j] != fin[j] for j in range(n)], float)
F["확정 margin 최소(-)"] = -np.where(cont, np.take_along_axis(MG, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0], 1e9).min(1)
rk = lambda x: (np.argsort(np.argsort(x)) + 0.5) / len(x)
F["조합(TRE식+다른step비율+확정logprob최소)"] = rk(F["TRE식 (늦게 확정 x 불확실)"]) + rk(F["최종답과 다른 step 비율"]) + rk(F["확정 logprob 최소(-)"])
def within(s, idx):
    g = collections.defaultdict(list)
    for j in idx: g[fid7[j]].append(j)
    a = [roc_auc_score(~corr[v], s[v]) for v in g.values() if 0 < corr[v].sum() < len(v)]; return round(float(np.mean(a)), 3)
mi = np.where(ism)[0]; det = {}
for nm, s in F.items():
    det[nm] = {"T3 1~4/5 안": round(float(roc_auc_score(~corr[mi], s[mi])), 3), "T3 같은사실": within(s, mi), "T2 (가중)": round(float(roc_auc_score(mw, s, sample_weight=W)), 3),
               "1~4/5오답 대 5/5": round(float(roc_auc_score(np.r_[np.ones(mw.sum()), np.zeros(isf.sum())], np.r_[s[mw], s[isf]])), 3)}
    print(nm, det[nm], flush=True)
O["생성뒤_감지"] = det
# ---- 두 번 생성 모의 (S 행).  감지 점수 상위 q 를 다시 생성 (조향).  문턱 q 는 다른 겹에서 1~4/5 가중 정답률 최대로.
E = collections.defaultdict(dict); SET = {}
for f in ("results_e8.jsonl", "results_e9.jsonl"):
    for l in open(f"{OUT}/{f}"):
        d = json.loads(l); c = d["cond"] if f.endswith("e9.jsonl") else f"[e8]{d['cond']}"; E[c][d["row"]] = d["new_type"] == "correct"; SET[d["row"]] = d["set"]
pos7 = {int(i): j for j, i in enumerate(RR)}; S = [i for _, i in set_S() if i in pos7]; sets = np.array([SET[i] for i in S]); gS = grp[S]
WS = {"broken": 310 / 310, "entity": 2749 / 300, "correct": 2875 / 300, "zero": 44110 / 300, "five": 5165 / 300}; ws = np.array([WS[x] for x in sets]); mS = np.isin(sets, ["broken", "entity", "correct"])
first = np.array([corr[pos7[i]] for i in S]); wacc = lambda x, m: float(100 * (x * ws)[m].sum() / ws[m].sum())
O["표본"] = dict(S=len(S), 첫생성_1to4=round(wacc(first, mS), 2), E9base_1to4=round(wacc(np.array([E['base'][i] for i in S]), mS), 2), 첫생성과_E9base_일치=round(100 * float(np.mean(first == np.array([E['base'][i] for i in S]))), 1))
print(O["표본"], flush=True)
tp = {}
for cond in ("B c0.5", "B c1", "Brel c1", "ACE t27", "B c1 s0-3", "[e8]maskgit_plus", "[e8]entropy"):
    redo = np.array([E[cond][i] for i in S])
    for snm in ("조합(TRE식+다른step비율+확정logprob최소)", "최종답과 다른 step 비율", "TRE식 (늦게 확정 x 불확실)"):
        sc = np.array([F[snm][pos7[i]] for i in S]); sel = np.zeros(len(S), bool); qs = []
        for g in range(NF):
            tr, te = gS != g, gS == g; bq, bv = 0.0, -1e9
            for q in np.linspace(0, 1, 51):
                th = np.quantile(sc[tr], 1 - q) if q > 0 else np.inf; v = wacc(np.where(sc >= th, redo, first), tr & mS)
                if v > bv: bv, bq = v, q
            th = np.quantile(sc[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = sc[te] >= th; qs.append(round(float(bq), 2))
        p = np.where(sel, redo, first); fix = (~first & p & mS); brk = (first & ~p & mS); tw = ws[mS].sum()
        tp[f"{cond} | {snm}"] = {"1~4/5": round(wacc(p, mS), 2), "차이": round(wacc(p, mS) - wacc(first, mS), 2), "고침": round(100 * float(ws[fix].sum() / tw), 1), "깨짐": round(100 * float(ws[brk].sum() / tw), 1),
                                 "5/5": round(100 * float(p[sets == "five"].mean()), 1), "다시생성_비율q": qs}
    orc = np.where(~first, redo, first); tp[f"{cond} | 오라클"] = {"1~4/5": round(wacc(orc, mS), 2), "차이": round(wacc(orc, mS) - wacc(first, mS), 2)}
    tp[f"{cond} | 모두 다시"] = {"1~4/5": round(wacc(redo, mS), 2), "차이": round(wacc(redo, mS) - wacc(first, mS), 2), "5/5": round(100 * float(redo[sets == "five"].mean()), 1)}
O["두번생성"] = tp
for kx, v in tp.items(): print(kx, v, flush=True)
# 부트스트랩 CI (최선 몇 개)
rng = np.random.default_rng(0); POS = {st: np.where(sets == st)[0] for st in WS}
O["메모"] = "사후 탐색.  문턱은 겹밖.  첫 생성 정답률은 E7 실행 기준."
json.dump(O, open(f"{OUT}/e11_twopass.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DONE", flush=True)
