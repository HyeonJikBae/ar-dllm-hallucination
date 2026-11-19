"""E10 (PREREG.md, CPU): 학습 없는 감지.  E7 수집 결과 사용, E9 가 있으면 게이팅까지.  usage: python followup_mk/e10_detect.py [--pilot]"""
import collections, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "followup_mk")
from common import rows, grp, ty, k, fid, mid, unit, NF, OUT, MODEL, REV, extract_answer, is_correct, set_S
PIL = "--pilot" in sys.argv; D7 = f"{OUT}/e7{'_pilot' if PIL else ''}"; O = {}
R7 = np.load(f"{D7}/rows.npy"); cfg = json.load(open(f"{D7}/config.json")); done = np.load(f"{D7}/done.npy")
meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
keep = np.array([done[j] and int(R7[j]) in meta for j in range(len(R7))]); J = np.where(keep)[0]; RR = R7[J]
print(f"E7 행 {len(R7)}, 사용 {len(J)}", flush=True)
HID = np.load(f"{D7}/hid.npy", mmap_mode="r"); IDS = np.load(f"{D7}/ids.npy", mmap_mode="r")[J]; LP = np.load(f"{D7}/lp.npy", mmap_mode="r")[J].astype(np.float32)
MG = np.load(f"{D7}/mg.npy", mmap_mode="r")[J].astype(np.float32); MK = np.load(f"{D7}/mask.npy", mmap_mode="r")[J]
kk = k[RR]; g7 = grp[RR]; corr = np.array([meta[int(i)]["new_type"] == "correct" for i in RR]); fid7 = fid[RR]
isz, isf, ism = kk == 0, kk == 5, (kk >= 1) & (kk <= 4)
W = np.where(isz, 44110 / max(isz.sum(), 1), np.where(isf, 5165 / max(isf.sum(), 1), 1.0))
T = {"T1 사실(1~4/5 대 나머지)": ism.astype(int), "T2 질문(아는데 틀림 대 나머지)": (ism & ~corr).astype(int)}
O["표본"] = dict(n=len(RR), mid=int(ism.sum()), zero=int(isz.sum()), five=int(isf.sum()), mid_이번정답률=round(100 * corr[ism].mean(), 1), 저장라벨일치=round(100 * np.mean([(ty[i] == "correct") == c for i, c in zip(RR[ism], corr[ism])]), 1))
print(O["표본"], flush=True)
def auc(y, s, w=None, idx=None):
    if idx is not None: y, s, w = y[idx], s[idx], (None if w is None else w[idx])
    if len(np.unique(y)) < 2: return np.nan
    return roc_auc_score(y, s, sample_weight=w)
def within(score, idx):
    gg = collections.defaultdict(list)
    for j in idx: gg[fid7[j]].append(j)
    a = [roc_auc_score(~corr[v], score[v]) for v in gg.values() if 0 < corr[v].sum() < len(v)]
    return float(np.mean(a)) if a else np.nan
mi = np.where(ism)[0]
# ---- 궤적 점수 (학습 없음).  부호: 높을수록 '정답 / 앎'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SP = set(tok.all_special_ids)
lastm = np.where(MK, np.arange(MK.shape[1])[None, :, None], -1).max(1)                 # 칸마다 마지막으로 가려져 있던 step
committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]       # 칸마다 확정된 토큰
def x0hat(s): return np.where(MK[:, s], IDS[:, s], committed)
def dec(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SP), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
fin = [dec(x0hat(31)[j]) for j in range(len(RR))]; ok_rec = np.mean([a == meta[int(i)]["answer"] for a, i in zip(fin, RR)]); O["궤적_복원_일치"] = round(100 * float(ok_rec), 2); print("최종 답 복원 일치", O["궤적_복원_일치"], flush=True)
SS = list(range(0, 32, 2)) + [31]; inter = {s: [dec(x0hat(s)[j]) for j in range(len(RR))] for s in SS}
gold = lambda i: {"gold_value": rows[i]["gold_value"], "gold_aliases": rows[i]["gold_aliases"]}
cache = {}
def ic(i, a):
    key = (i, a)
    if key not in cache: cache[key] = bool(is_correct({"model_answer": a, **gold(i)}))
    return cache[key]
icm = {s: np.array([ic(int(i), inter[s][j]) for j, i in enumerate(RR)]) for s in SS}
anymid = np.any([icm[s] for s in SS if s < 31], 0); lost = anymid & ~corr
O["진동"] = {nm: dict(n=int(ix.sum()), 중간에_정답_나옴=round(100 * anymid[ix].mean(), 1), 나왔다가_사라짐=round(100 * lost[ix].mean(), 1)) for nm, ix in (("0/5", isz), ("1~4/5 정답", ism & corr), ("1~4/5 오답", ism & ~corr), ("5/5", isf))}
print("진동", O["진동"], flush=True)
SC = {}
for s in (0, 4, 8, 31):
    SC[f"첫칸 logprob s{s}"] = LP[:, s, 0]; SC[f"첫4칸 logprob s{s}"] = LP[:, s, :4].mean(1); SC[f"첫칸 margin s{s}"] = MG[:, s, 0]; SC[f"첫4칸 margin s{s}"] = MG[:, s, :4].mean(1)
    if s > 0:
        SC[f"첫칸 확정됨 s{s}"] = (~MK[:, s, 0]).astype(float); SC[f"흔들림(-) s{s}"] = -(IDS[:, 1:s + 1, :4] != IDS[:, :s, :4]).sum((1, 2)).astype(float)
        SC[f"중간답 종류(-) s{s}"] = -np.array([len({inter[t][j] for t in SS if t <= s}) for j in range(len(RR))], float)
# ---- 평균 차이 투영 (다른 겹에서 방향).  높을수록 앎 / 정답
LAY, SV, PS = cfg["layers"], cfg["steps"], cfg["positions"]; rel = np.array([rows[i]["relation"] for i in RR]); PROJ = {}
for si, s in enumerate(SV):
    for li, l_ in enumerate(LAY):
        for pi, p_ in enumerate(PS):
            Xh = np.asarray(HID[J, si, li, pi], dtype=np.float32); pk = np.zeros(len(RR)); pb = np.zeros(len(RR)); mm_ = np.zeros(len(RR))
            for g in range(NF):
                tr = g7 != g; te = ~tr; ds = []
                for r_ in np.unique(rel):
                    a = tr & isf & (rel == r_); b = tr & isz & (rel == r_)
                    if a.sum() >= 10 and b.sum() >= 10: ds.append(Xh[a].mean(0) - Xh[b].mean(0))
                K = unit(np.mean(ds, 0)) if ds else unit(Xh[tr & isf].mean(0) - Xh[tr & isz].mean(0))
                byf = collections.defaultdict(lambda: collections.defaultdict(list))
                for j in np.where(tr & ism)[0]: byf[fid7[j]]["c" if corr[j] else "w"].append(j)
                dd = [Xh[v["c"]].mean(0) - Xh[v["w"]].mean(0) for v in byf.values() if v["c"] and v["w"]]
                Bw = unit(np.mean(dd, 0)) if dd else unit(Xh[tr & ism & corr].mean(0) - Xh[tr & ism & ~corr].mean(0)); pk[te] = Xh[te] @ K; pb[te] = Xh[te] @ Bw; mm_[te] = np.median(Xh[tr & ism] @ K)
            tag = f"L{l_} {p_} s{s}"; PROJ[f"K {tag}"] = pk; PROJ[f"Bw {tag}"] = pb; PROJ[f"가운데K {tag}"] = -np.abs(pk - mm_)
    print(f"  투영 step {s} 끝", flush=True)
SC.update(PROJ)
rk = lambda x: (np.argsort(np.argsort(x)) + 0.5) / len(x)
# T2 조합 (사용자 제안 두 단계: 사실 층위 '가운데' x 질문 층위 '오답 쪽')
for s in (0, 4, 8):
    for l_ in LAY:
        for p_ in PS:
            mkey = f"가운데K L{l_} {p_} s{s}"
            if mkey in SC:
                SC[f"조합[가운데K x 첫4칸logprob(-)] L{l_} {p_} s{s}"] = rk(SC[mkey]) * rk(-SC[f"첫4칸 logprob s{s}"])
                SC[f"조합[가운데K x Bw(-)] L{l_} {p_} s{s}"] = rk(SC[mkey]) * rk(-SC[f"Bw L{l_} {p_} s{s}"])
step_of = lambda nm: int(nm.split(" s")[-1])
def table(y, sign_for_target, wts):
    res = {}
    for nm, s in SC.items():
        sc = s if sign_for_target(nm) else -s
        res[nm] = auc(y, sc, wts)
    return res
# T1: 가운데 점수는 그대로, 나머지 점수는 양방향이 의미 없으므로 가운데/조합만 1 순위 후보로.  T2: 오답 쪽이 양성 -> '앎/정답' 점수는 부호 반대, 조합은 그대로
res = {}
res["T1"] = {nm: auc(T["T1 사실(1~4/5 대 나머지)"], s, W) for nm, s in SC.items() if nm.startswith("가운데")}
POSN = lambda nm: 1 if nm.startswith(("조합", "가운데")) else -1
res["T2"] = {nm: auc(T["T2 질문(아는데 틀림 대 나머지)"], POSN(nm) * s, W) for nm, s in SC.items()}
res["T3"] = {nm: auc((~corr).astype(int), -s, None, mi) for nm, s in SC.items() if not nm.startswith(("가운데", "조합"))}
res["T3_같은사실"] = {nm: within(-SC[nm], mi) for nm in res["T3"]}
def nested(target, y, sgn, idx, wts, early_only):
    """겹마다 다른 겹에서 최선 점수를 고르고 남은 겹에서 평가."""
    cands = [nm for nm in res[target] if (not early_only or step_of(nm) <= 8)]; out = []
    for g in range(NF):
        tr = np.intersect1d(idx, np.where(g7 != g)[0]); te = np.intersect1d(idx, np.where(g7 == g)[0])
        best = max(cands, key=lambda nm: np.nan_to_num(auc(y, sgn(nm) * SC[nm], wts, tr), nan=0)); out.append((best, auc(y, sgn(nm := best) * SC[best], wts, te)))
    return dict(평균=round(float(np.nanmean([a for _, a in out])), 3), 고른점수=[b for b, _ in out])
allix = np.arange(len(RR))
O["감지"] = {}
for tname, key, y, sg, idx, wts in (("T1 사실", "T1", T["T1 사실(1~4/5 대 나머지)"], lambda nm: 1, allix, W), ("T2 질문", "T2", T["T2 질문(아는데 틀림 대 나머지)"], POSN, allix, W),
                                     ("T3 1~4/5 안", "T3", (~corr).astype(int), lambda nm: -1, mi, None)):
    for early in (True, False):
        cand = {nm: v for nm, v in res[key].items() if (not early or step_of(nm) <= 8)}
        top = sorted(cand.items(), key=lambda x: -np.nan_to_num(x[1]))[:8]
        O["감지"][f"{tname} {'step<=8' if early else '사후 포함'}"] = dict(상위=[(nm, round(float(v), 3)) for nm, v in top], 겹밖최선=nested(key, y, sg, idx, wts, early))
    print(tname, O["감지"][f"{tname} step<=8"], flush=True)
O["감지"]["T3 같은사실 step<=8 상위"] = [(nm, round(float(v), 3)) for nm, v in sorted(((nm, v) for nm, v in res["T3_같은사실"].items() if step_of(nm) <= 8), key=lambda x: -np.nan_to_num(x[1]))[:8]]
best = lambda key: max([nm for nm in res[key] if step_of(nm) <= 8], key=lambda nm: np.nan_to_num(res[key][nm]))
O["판정"] = dict(H10_1=bool(res["T1"][best("T1")] >= .75), H10_2=bool(res["T2"][best("T2")] >= .80), H10_3=bool(res["T3_같은사실"][max([nm for nm in res["T3_같은사실"] if step_of(nm) <= 8], key=lambda nm: np.nan_to_num(res["T3_같은사실"][nm]))] >= .70))
# ---- 참고: 학습한 probe (상한 확인용)
ref = {}
for s in (0, 8):
    for l_ in (12, 20, 27):
        Xh = np.asarray(HID[J, SV.index(s), LAY.index(l_), 0], dtype=np.float32); out = {}
        for tn, y, idx in (("T1", T["T1 사실(1~4/5 대 나머지)"], allix), ("T2", T["T2 질문(아는데 틀림 대 나머지)"], allix), ("T3", (~corr).astype(int), mi)):
            sc = np.zeros(len(RR))
            for g in range(NF):
                tr = np.intersect1d(idx, np.where(g7 != g)[0]); te = np.intersect1d(idx, np.where(g7 == g)[0]); mu = Xh[tr].mean(0); sd = Xh[tr].std(0) + 1e-6
                if len(np.unique(y[tr])) < 2: continue
                clf = LogisticRegression(C=0.01, max_iter=1000).fit((Xh[tr] - mu) / sd, y[tr], sample_weight=None if tn == "T3" else W[tr]); sc[te] = clf.decision_function((Xh[te] - mu) / sd)
            out[tn] = round(float(auc(y, sc, None if tn == "T3" else W, idx)), 3)
            if tn == "T3": out["T3_같은사실"] = round(within(sc, mi), 3)
        ref[f"L{l_} prompt_end s{s}"] = out; print("probe", s, l_, out, flush=True)
O["참고_학습probe"] = ref
# ---- 게이팅 (E9)
f9 = f"{OUT}/results_e9.jsonl"
if os.path.exists(f9) and not PIL:
    E9 = collections.defaultdict(dict); SET = {}
    for l in open(f9): d = json.loads(l); E9[d["cond"]][d["row"]] = d["new_type"] == "correct"; SET[d["row"]] = d["set"]
    S = [i for _, i in set_S() if i in E9["base"]]; pos7 = {int(i): j for j, i in enumerate(RR)}; S = [i for i in S if i in pos7]
    WS = {"broken": 310 / 310, "entity": 2749 / 300, "correct": 2875 / 300, "zero": 44110 / 300, "five": 5165 / 300}
    ws = np.array([WS[SET[i]] for i in S]); m_ = np.array([SET[i] in ("broken", "entity", "correct") for i in S]); gS = grp[S]
    base = np.array([E9["base"][i] for i in S]); late = np.array([E9["B c1 s8-31"][i] for i in S])
    wacc = lambda x, msk: float(100 * (x * ws)[msk].sum() / ws[msk].sum())
    t2c = [nm for nm in res["T2"] if step_of(nm) <= 8]
    def policy(sel): return np.where(sel, late, base)
    gate = np.zeros(len(S), bool); chosen = []
    for g in range(NF):
        tr = gS != g; te = gS == g
        trrows = np.array([pos7[i] for i in np.array(S)[tr]]); bestnm = max(t2c, key=lambda nm: np.nan_to_num(auc(T["T2 질문(아는데 틀림 대 나머지)"][trrows], (POSN(nm) * SC[nm])[trrows], W[trrows])))
        sc = np.array([POSN(bestnm) * SC[bestnm][pos7[i]] for i in S]); best_q, best_v = 0, -1e9
        for q in np.linspace(0, 1, 51):
            th = np.quantile(sc[tr], 1 - q) if q > 0 else np.inf; sel = sc >= th; v = wacc(policy(sel), tr & m_)
            if v > best_v: best_v, best_q = v, q
        th = np.quantile(sc[tr], 1 - best_q) if best_q > 0 else np.inf; gate[te] = sc[te] >= th; chosen.append((bestnm, round(best_q, 2)))
    oracle = np.where(~base & m_, late, base)
    O["게이팅"] = {"1~4/5 가중 정답률": dict(base=round(wacc(base, m_), 1), 모두_s8_31=round(wacc(late, m_), 1), 모두_B_c1=round(wacc(np.array([E9['B c1'][i] for i in S]), m_), 1), 모두_B_c0_5=round(wacc(np.array([E9['B c0.5'][i] for i in S]), m_), 1),
                   감지게이트=round(wacc(policy(gate), m_), 1), 오라클_s8_31=round(wacc(oracle, m_), 1)),
                   "전체 가중 정답률": dict(base=round(wacc(base, np.ones(len(S), bool)), 2), 감지게이트=round(wacc(policy(gate), np.ones(len(S), bool)), 2), 모두_s8_31=round(wacc(late, np.ones(len(S), bool)), 2)),
                   "고른점수_q": chosen, "민 비율": round(100 * gate.mean(), 1)}
    gg = O["게이팅"]["1~4/5 가중 정답률"]; O["판정"]["H_G"] = bool(gg["감지게이트"] - gg["base"] >= 2 and gg["감지게이트"] > max(gg["모두_B_c1"], gg["모두_B_c0_5"]))
    print("게이팅", O["게이팅"], flush=True)
json.dump(O, open(f"{OUT}/e10_summary{'_pilot' if PIL else ''}.json", "w"), ensure_ascii=False, indent=1)
print("판정", O["판정"]); print("DONE", flush=True)
