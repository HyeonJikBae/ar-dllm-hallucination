"""E15 (PREREG.md, CPU): 순이득 식, 되감기 조향 오프라인 모의, '무엇으로든 고쳐지는' 상한, 시간 다수결 기준선, E11 CI.
usage: python followup_mk/e15_calc.py   (repo 루트)"""
import collections, itertools, json, sys
import numpy as np
from scipy.stats import norm as N
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "followup_mk")
from common import rows, grp, k, fid, unit, NF, OUT, MODEL, REV, POP, extract_answer, is_correct, set_S
O = {}; rng0 = np.random.default_rng(0); NB = 10000
# ================= 공용: E7 궤적, S 행, 조건 결과
D7 = f"{OUT}/e7"; R7 = np.load(f"{D7}/rows.npy"); done = np.load(f"{D7}/done.npy"); cfg = json.load(open(f"{D7}/config.json")); meta = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); meta[d["row"]] = d
J = np.where([done[j] and int(R7[j]) in meta for j in range(len(R7))])[0]; RR = R7[J]; n = len(RR)
IDS = np.asarray(np.load(f"{D7}/ids.npy", mmap_mode="r")[J]); LP = np.asarray(np.load(f"{D7}/lp.npy", mmap_mode="r")[J]).astype(np.float32)
MG = np.asarray(np.load(f"{D7}/mg.npy", mmap_mode="r")[J]).astype(np.float32); MK = np.asarray(np.load(f"{D7}/mask.npy", mmap_mode="r")[J])
kk = k[RR]; g7 = grp[RR]; corr = np.array([meta[int(i)]["new_type"] == "correct" for i in RR]); fid7 = fid[RR]; rel7 = np.array([rows[i]["relation"] for i in RR])
isz, isf, ism = kk == 0, kk == 5, (kk >= 1) & (kk <= 4)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True); SPL = sorted(tok.all_special_ids); SPS = set(SPL)
lastm = np.where(MK, np.arange(32)[None, :, None], -1).max(1); cstep = lastm + 1
committed = np.take_along_axis(IDS, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
isSP = np.isin(committed, SPL); Lc = np.maximum(np.where(isSP.any(1), isSP.argmax(1), 32), 1); cont = np.arange(32)[None, :] < Lc[:, None]
lp_c = np.take_along_axis(LP, np.clip(lastm, 0, None)[:, None, :], 1)[:, 0]
x0 = lambda s: np.where(MK[:, s], IDS[:, s], committed)
def dc(t):
    t = list(t); t = t[: next((q for q, v in enumerate(t) if v in SPS), len(t))]; return extract_answer(tok.decode(t, skip_special_tokens=True))
ANS = [[dc(r) for r in x0(s)] for s in range(32)]; fin = ANS[31]
print("궤적 복원 일치", np.mean([a == meta[int(i)]["answer"] for a, i in zip(fin, RR)]), flush=True)
_gc = {}
def ok(i, a):
    key = (int(i), a)
    if key not in _gc: _gc[key] = bool(is_correct({"model_answer": a, "gold_value": rows[i]["gold_value"], "gold_aliases": rows[i]["gold_aliases"]}))
    return _gc[key]
rk = lambda x: (np.argsort(np.argsort(x, kind="stable"), kind="stable") + 0.5) / len(x)
# 점수 (높을수록 '틀림')
F = {}
F["최종답과 다른 step 비율"] = np.array([np.mean([ANS[s][j] != fin[j] for s in range(31)]) for j in range(n)])
F["TRE식 (늦게 확정 x 불확실)"] = ((-lp_c) * (cstep / 32.0) * cont).sum(1) / cont.sum(1)
F["확정 logprob 최소(-)"] = -np.where(cont, lp_c, 0).min(1)
F["조합(TRE식+다른step비율+확정logprob최소)"] = rk(F["TRE식 (늦게 확정 x 불확실)"]) + rk(F["최종답과 다른 step 비율"]) + rk(F["확정 logprob 최소(-)"])
F["첫4칸 logprob s0(-)"] = -LP[:, 0, :4].mean(1)
# Bw L20 s8 (겹밖, e10 과 같은 식).  높을수록 정답 -> 부호 반대
HID = np.load(f"{D7}/hid.npy", mmap_mode="r"); Xh = np.asarray(HID[J, cfg["steps"].index(8), cfg["layers"].index(20), 0], dtype=np.float32); pb = np.zeros(n)
for g in range(NF):
    tr = g7 != g; byf = collections.defaultdict(lambda: collections.defaultdict(list))
    for j in np.where(tr & ism)[0]: byf[fid7[j]]["c" if corr[j] else "w"].append(j)
    Bw = unit(np.mean([Xh[v["c"]].mean(0) - Xh[v["w"]].mean(0) for v in byf.values() if v["c"] and v["w"]], 0)); pb[~tr] = Xh[~tr] @ Bw
F["Bw L20 s8(-)"] = -pb; del Xh
# 되감기용 신호 (step <= 4 만)
F["[s<=4] 중간답 종류 s0-4"] = np.array([len({ANS[s][j] for s in range(5)}) for j in range(n)], float)
F["[s<=4] s0 답 != s4 답"] = np.array([ANS[0][j] != ANS[4][j] for j in range(n)], float) + 1e-3 * rk(-LP[:, 4, :4].mean(1))
F["[s<=4] 첫4칸 logprob s0(-)"] = -LP[:, 0, :4].mean(1)
F["[s<=4] 첫4칸 logprob s4(-)"] = -LP[:, 4, :4].mean(1)
F["[s<=4] 흔들림 s4"] = (IDS[:, 1:5, :4] != IDS[:, :4, :4]).sum((1, 2)).astype(float) + 1e-3 * rk(-LP[:, 4, :4].mean(1))
F["[s<=4] 첫칸 미확정 s4"] = MK[:, 4, 0].astype(float) + 1e-3 * rk(-LP[:, 4, :4].mean(1))
F["[s<=4] 조합(종류+logprob s4+흔들림)"] = rk(F["[s<=4] 중간답 종류 s0-4"]) + rk(F["[s<=4] 첫4칸 logprob s4(-)"]) + rk(F["[s<=4] 흔들림 s4"])
E = collections.defaultdict(dict); SET = {}
for f in ("results_e8.jsonl", "results_e9.jsonl"):
    for l in open(f"{OUT}/{f}"):
        d = json.loads(l); c = d["cond"] if f.endswith("e9.jsonl") else f"[e8]{d['cond']}"; E[c][d["row"]] = d["new_type"] == "correct"; SET[d["row"]] = d["set"]
pos7 = {int(i): j for j, i in enumerate(RR)}; S = [i for _, i in set_S() if i in pos7]; Sa = np.array(S); sets = np.array([SET[i] for i in S]); gS = grp[Sa]; jS = np.array([pos7[i] for i in S])
WS = {"broken": 310 / 310, "entity": 2749 / 300, "correct": 2875 / 300, "zero": 44110 / 300, "five": 5165 / 300}; ws = np.array([WS[x] for x in sets]); mS = np.isin(sets, ["broken", "entity", "correct"])
first = corr[jS]; assert np.mean(first == np.array([E["base"][i] for i in S])) > .99
wacc = lambda x, m: float(100 * (x * ws)[m].sum() / ws[m].sum())
MIDSETS = ["broken", "entity", "correct"]; POS = {st: np.where(sets == st)[0] for st in WS}
BIX = {st: rng0.integers(0, len(POS[st]), (NB, len(POS[st]))) for st in WS}
CW = {st: WS[st] * len(POS[st]) / sum(WS[s2] * len(POS[s2]) for s2 in MIDSETS) for st in MIDSETS}
def boot_gain(p):
    """1~4/5 가중 정답률 이득 (p - first) 의 점추정, 95% CI, 단측 p (이득 <= 0)."""
    d = (p.astype(float) - first.astype(float)); bs = sum(CW[st] * d[POS[st]][BIX[st]].mean(1) for st in MIDSETS) * 100
    est = sum(CW[st] * d[POS[st]].mean() for st in MIDSETS) * 100
    return [round(float(est), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)], float((np.sum(bs <= 0) + 1) / (NB + 1))
def five(p): return round(100 * float(p[sets == "five"].mean()), 1)
def foldout_gate(sc, redo, qgrid=np.linspace(0, 1, 51)):
    sel = np.zeros(len(S), bool); qs = []
    for g in range(NF):
        tr, te = gS != g, gS == g; bq, bv = 0.0, -1e9
        for q in qgrid:
            th = np.quantile(sc[tr], 1 - q) if q > 0 else np.inf; v = wacc(np.where(sc >= th, redo, first), tr & mS)
            if v > bv: bv, bq = v, q
        th = np.quantile(sc[tr], 1 - bq) if bq > 0 else np.inf; sel[te] = sc[te] >= th; qs.append(round(float(bq), 2))
    return sel, qs
p0 = wacc(first, mS) / 100; O["기저"] = dict(S=len(S), 첫생성_1to4=round(100 * p0, 2), 첫생성_5of5=five(first))
# ================= 1. 순이득 식
CONDS = ["B c0.5", "B c1", "Brel c1", "ACE t27", "[e8]maskgit_plus", "[e8]entropy"]
wW = ws * mS * ~first; wC = ws * mS * first
fb = {}
for c in CONDS:
    x = np.array([E[c][i] for i in S]); fb[c] = dict(f=float((wW * x).sum() / wW.sum()), b=float((wC * ~x).sum() / wC.sum()))
def binormal_curve(a, f, b, p=p0, ngrid=2001):
    d = np.sqrt(2) * N.ppf(min(a, 1 - 1e-9)); t = np.linspace(-8, 8 + d, ngrid); tpr = N.sf(t - d); fpr = N.sf(t)
    gain = 100 * ((1 - p) * f * tpr - p * b * fpr); frac = (1 - p) * tpr + p * fpr; j = int(np.argmax(gain)); return float(gain[j]), float(frac[j]), gain, frac
def need_auc(target, f, b):
    if 100 * (1 - p0) * f <= target: return None
    lo, hi = 0.5, 0.99999
    for _ in range(60):
        mid_ = (lo + hi) / 2
        if binormal_curve(mid_, f, b)[0] >= target: hi = mid_
        else: lo = mid_
    return round(hi, 3)
sec1 = {"p(1~4/5 기저 정답)": round(p0, 4), "조건": {}}
for c in CONDS:
    f, b = fb[c]["f"], fb[c]["b"]; row = dict(고침률f=round(100 * f, 1), 깨짐률b=round(100 * b, 1), 오라클이득=round(100 * (1 - p0) * f, 2), 모두다시이득=round(100 * ((1 - p0) * f - p0 * b), 2))
    for tg in (3, 5, 8): row[f"+{tg} 필요 AUROC"] = need_auc(tg, f, b)
    row["곡선(AUROC: 최대이득, 다시생성비율)"] = {str(a): [round(binormal_curve(a, f, b)[0], 2), round(100 * binormal_curve(a, f, b)[1], 1)] for a in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)}
    sec1["조건"][c] = row; print("1.", c, {k_: v for k_, v in row.items() if not k_.startswith("곡선")}, flush=True)
# 경험 곡선 대 이항정규 예측, 독립 가정 점검
emp = {}
for snm in ("최종답과 다른 step 비율", "첫4칸 logprob s0(-)", "Bw L20 s8(-)"):
    sc = F[snm][jS]; mi = np.where(mS)[0]; a_emp = float(roc_auc_score(~first[mi], sc[mi], sample_weight=ws[mi])); emp[snm] = {"S 1~4/5 가중 AUROC(오답 양성)": round(a_emp, 3)}
    order = mi[np.argsort(-sc[mi], kind="stable")]; cw = np.cumsum(ws[order]) / ws[mi].sum()
    for c in ("B c0.5", "Brel c1", "B c1"):
        x = np.array([E[c][i] for i in S]); f, b = fb[c]["f"], fb[c]["b"]; curve = {}
        for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            sel = np.zeros(len(S), bool); sel[order[cw <= q + 1e-9]] = True; curve[str(q)] = round(wacc(np.where(sel, x, first), mS) - 100 * p0, 2)
        g_bn, fr_bn, gg, ff = binormal_curve(a_emp, f, b); bnq = {str(q): round(float(np.interp(q, ff[::-1], gg[::-1])), 2) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)}
        sel_fo, qs = foldout_gate(sc, x)
        emp[snm][c] = {"경험(다시생성비율:이득, 표본 안)": curve, "이항정규 예측": bnq, "이항정규 최대": [round(g_bn, 2), round(100 * fr_bn, 1)], "경험 최대(표본안)": max(curve.values()), "겹밖 문턱": boot_gain(np.where(sel_fo, x, first))[0]}
        # 독립 가정: 점수 3분위별 f, b
        tq = np.quantile(sc[mi], [1 / 3, 2 / 3]); bins = np.digitize(sc, tq); ind = {}
        for t_, nm_ in ((0, "낮음(정답쪽)"), (1, "중간"), (2, "높음(오답쪽)")):
            m_ = mS & (bins == t_); w_ = ws * m_ * ~first; c_ = ws * m_ * first
            ind[nm_] = dict(f=round(100 * float((w_ * x).sum() / max(w_.sum(), 1e-9)), 1), b=round(100 * float((c_ * ~x).sum() / max(c_.sum(), 1e-9)), 1), 오답비율=round(100 * float(w_.sum() / (ws * m_).sum()), 1))
        emp[snm][c]["3분위별 f,b"] = ind
    print("1. 경험", snm, json.dumps(emp[snm], ensure_ascii=False)[:1500], flush=True)
sec1["경험곡선"] = emp; O["1_순이득식"] = sec1
# ================= 2. 되감기 조향 (step <= 4 신호로 고르고 B c0.5 로 처음부터)
sec2 = {}; early = [nm for nm in F if nm.startswith("[s<=4]")]
for c in ("B c0.5", "ACE t27", "[e8]maskgit_plus", "Brel c1"):
    x = np.array([E[c][i] for i in S]); res = {}
    for snm in early:
        sc = F[snm][jS]; sel, qs = foldout_gate(sc, x); p = np.where(sel, x, first); (g_, lo, hi), pv = boot_gain(p)
        fl = float((ws * sel)[mS].sum() / ws[mS].sum())
        res[snm] = dict(이득CI=[g_, lo, hi], 단측p=round(pv, 4), 고침=round(100 * float((ws * (~first & p))[mS].sum() / ws[mS].sum()), 1), 깨짐=round(100 * float((ws * (first & ~p))[mS].sum() / ws[mS].sum()), 1),
                        five=five(p), 다시생성비율_1to4=round(100 * fl, 1), 다시생성비율_5of5=round(100 * float(sel[sets == "five"].mean()), 1), 평균생성비용_1to4=round(1 + 0.125 * fl, 3), 겹별q=qs)
    # 겹밖에서 신호까지 고르기 (이중 겹밖)
    sel = np.zeros(len(S), bool); pick = []
    for g in range(NF):
        tr, te = gS != g, gS == g; best = None
        for snm in early:
            sc = F[snm][jS]
            for q in np.linspace(0, 1, 51):
                th = np.quantile(sc[tr], 1 - q) if q > 0 else np.inf; v = wacc(np.where(sc >= th, x, first), tr & mS)
                if best is None or v > best[0]: best = (v, snm, q)
        _, snm, q = best; sc = F[snm][jS]; th = np.quantile(sc[tr], 1 - q) if q > 0 else np.inf; sel[te] = sc[te] >= th; pick.append((snm, round(float(q), 2)))
    p = np.where(sel, x, first); (g_, lo, hi), pv = boot_gain(p)
    res["이중겹밖(신호도 겹밖 선택)"] = dict(이득CI=[g_, lo, hi], 단측p=round(pv, 4), five=five(p), 고른것=pick)
    res["모두 다시"] = dict(이득CI=boot_gain(x)[0], five=five(x)); res["오라클"] = dict(이득=round(wacc(np.where(~first, x, first), mS) - 100 * p0, 2))
    sec2[c] = res; print("2.", c, json.dumps(res, ensure_ascii=False)[:2000], flush=True)
# step<=4 신호의 감지력
sec2["감지력(S 1~4/5 가중 AUROC, 전체 1~4/5 같은사실)"] = {}
mi7 = np.where(ism)[0]
def within(s, idx):
    gg = collections.defaultdict(list)
    for j in idx: gg[fid7[j]].append(j)
    a = [roc_auc_score(~corr[v], s[v]) for v in gg.values() if 0 < corr[v].sum() < len(v)]; return round(float(np.mean(a)), 3)
for snm in early + ["최종답과 다른 step 비율"]:
    sec2["감지력(S 1~4/5 가중 AUROC, 전체 1~4/5 같은사실)"][snm] = [round(float(roc_auc_score(~first[mS], F[snm][jS][mS], sample_weight=ws[mS])), 3), round(float(roc_auc_score(~corr[mi7], F[snm][mi7])), 3), within(F[snm], mi7)]
print("2. 감지력", sec2["감지력(S 1~4/5 가중 AUROC, 전체 1~4/5 같은사실)"], flush=True)
O["2_되감기조향"] = sec2
# ================= 3. 무엇으로든 고쳐지는 상한
X3 = {}
for c in E:
    if c != "base": X3[f"E{'8' if c.startswith('[e8]') else '9'}:{c.replace('[e8]', '')}"] = {i: E[c][i] for i in S}
for f_, tag, basec in (("results_e3.jsonl", "E3", "zero"), ("results_e6.jsonl", "E6", "base"), ("results_e4.jsonl", "E4", "base")):
    for l in open(f"{OUT}/{f_}"):
        d = json.loads(l)
        if d["row"] not in pos7 or d["row"] not in SET: continue
        X3.setdefault(f"{tag}:{d['cond']}" if d["cond"] != basec else f"{tag}:기준선재실행", {})[d["row"]] = d["new_type"] == "correct"
wrong_mid = [j for j in range(len(S)) if mS[j] and not first[j]]
def fixed_by(cn): return np.array([X3[cn].get(S[j], False) for j in wrong_mid])
wwm = ws[wrong_mid]; wsum = wwm.sum(); setwm = sets[wrong_mid]
noise = [c for c in X3 if "기준선재실행" in c]; real = [c for c in X3 if c not in noise and "G0" not in c and "G1" not in c and "D0" not in c]
FX = {c: fixed_by(c) for c in X3}
cov = {c: np.array([S[j] in X3[c] for j in wrong_mid]) for c in X3}
def wr(m): return round(100 * float((wwm * m).sum() / wsum), 1)
anyfix = np.any([FX[c] for c in real], 0); anynoise = np.any([FX[c] for c in noise], 0)
allset = [c for c in real if cov[c].all()]
sec3 = {"1~4/5 오답 행 수(S)": len(wrong_mid), "무엇으로든 고쳐짐(전체 조건, 가중%)": wr(anyfix), "기준선 재실행만으로 정답(잡음, 가중%)": wr(anynoise),
        "세트별 무엇으로든": {st: round(100 * float(anyfix[setwm == st].mean()), 1) for st in ("broken", "entity", "correct")},
        "세트별 행수": {st: int((setwm == st).sum()) for st in ("broken", "entity", "correct")}}
single = sorted(((wr(FX[c]), c) for c in allset), reverse=True); sec3["단일 최선(모든 세트 덮는 조건)"] = single[:5]
best2 = max(((wr(FX[a] | FX[b]), (a, b)) for a, b in itertools.combinations(allset, 2)))
best3 = max(((wr(FX[a] | FX[b] | FX[c]), (a, b, c)) for a, b, c in itertools.combinations(allset, 3)))
sec3["둘 조합 최선"] = best2; sec3["셋 조합 최선"] = best3; sec3["모든 세트 덮는 조건 전부 합"] = wr(np.any([FX[c] for c in allset], 0))
# 특징 비교 (가중 없이, 1~4/5 오답 행)
rw = [S[j] for j in wrong_mid]; spop = np.array([POP[rows[i]["sample_id"]]["s_pop"] for i in rw], float); opop = np.array([POP[rows[i]["sample_id"]]["o_pop"] for i in rw], float)
glen = np.array([len(rows[i]["gold_value"]) for i in rw], float); kw = k[np.array(rw)]; relw = np.array([rows[i]["relation"] for i in rw])
cnt = np.sum([FX[c] for c in allset], 0)
def cmp(v):
    return {"고쳐짐 중앙값": round(float(np.median(v[anyfix])), 1), "안고쳐짐 중앙값": round(float(np.median(v[~anyfix])), 1), "AUROC(값 클수록 고쳐짐)": round(float(roc_auc_score(anyfix, v)), 3)}
feat = {"log s_pop": cmp(np.log1p(spop)), "log o_pop": cmp(np.log1p(opop)), "금답 글자수": cmp(glen), "k": cmp(kw.astype(float)),
        "k별 고쳐짐%": {int(x): round(100 * float(anyfix[kw == x].mean()), 1) for x in (1, 2, 3, 4)},
        "세트별(유형) 고쳐짐%": {st: round(100 * float(anyfix[setwm == st].mean()), 1) for st in ("broken", "entity", "correct")}}
rc = collections.Counter(relw); feat["관계별 고쳐짐%(n>=15)"] = {r_: [round(100 * float(anyfix[relw == r_].mean()), 1), int(rc[r_])] for r_ in sorted(rc, key=lambda r_: -rc[r_]) if rc[r_] >= 15}
feat["고쳐지는 조건 수 분포(모든세트 조건 중)"] = {int(v): int((cnt == v).sum()) for v in np.unique(cnt)}
sec3["특징"] = feat; O["3_고쳐지는상한"] = sec3; print("3.", json.dumps(sec3, ensure_ascii=False)[:3000], flush=True)
# ================= 4. 시간 다수결 (금 라벨 안 씀, 채점만 금 라벨)
norm_ = lambda a: " ".join(a.lower().strip().split())
def vote(j, wfun):
    c = collections.Counter(); first_seen = {}
    for s in range(32):
        a = norm_(ANS[s][j])
        if not a: continue
        c[a] += wfun(s); first_seen.setdefault(a, ANS[s][j])
    if not c: return fin[j]
    best = max(c.values()); cand = [a for a in c if c[a] == best]
    if norm_(fin[j]) in cand: return fin[j]
    return next(ANS[s][j] for s in range(31, -1, -1) if norm_(ANS[s][j]) in cand)
sec4 = {}
for vn, wfun in (("균등", lambda s: 1.0), ("늦은 step 선형 가중", lambda s: (s + 1) / 32.0), ("지수 가중(늦은 쪽)", lambda s: np.exp((s - 31) / 8.0))):
    va = [vote(j, wfun) for j in range(n)]; vc = np.array([ok(RR[j], va[j]) for j in range(n)])
    r_ = {}
    for nm, ix in (("1~4/5", ism), ("5/5", isf), ("0/5", isz)):
        r_[nm] = dict(기존=round(100 * float(corr[ix].mean()), 2), 다수결=round(100 * float(vc[ix].mean()), 2), 고침=int((~corr & vc & ix).sum()), 깨짐=int((corr & ~vc & ix).sum()), 답바뀜=int(np.sum([va[j] != fin[j] for j in np.where(ix)[0]])))
    dm = (vc[ism].astype(float) - corr[ism].astype(float)); bs = dm[rng0.integers(0, len(dm), (NB, len(dm)))].mean(1) * 100
    r_["1~4/5 이득CI(전체 5950행)"] = [round(float(dm.mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
    sec4[vn] = r_; print("4.", vn, r_, flush=True)
O["4_시간다수결"] = sec4
# ================= 5. E11 두 번 생성 CI 와 선택 낙관
sec5 = {}; E11S = ("조합(TRE식+다른step비율+확정logprob최소)", "최종답과 다른 step 비율", "TRE식 (늦게 확정 x 불확실)")
E11C = ("B c0.5", "B c1", "Brel c1", "ACE t27", "B c1 s0-3", "[e8]maskgit_plus", "[e8]entropy"); pvals = {}; SELS = {}
for c in E11C:
    x = np.array([E[c][i] for i in S])
    for snm in E11S:
        sel, qs = foldout_gate(F[snm][jS], x); p = np.where(sel, x, first); ci_, pv = boot_gain(p); key = f"{c} | {snm}"; SELS[key] = (x, F[snm][jS])
        sec5[key] = dict(이득CI=ci_, 단측p=round(pv, 4), five=five(p), q=qs); pvals[key] = pv
ks = sorted(pvals, key=lambda kx: pvals[kx]); m_ = len(ks); holm = {}; run = 0
for r_, kx in enumerate(ks): run = max(run, min(1.0, (m_ - r_) * pvals[kx])); holm[kx] = round(run, 4)
for kx in sec5: sec5[kx]["Holm_p"] = holm[kx]
# 이중 겹밖: 겹마다 21 조합 + 문턱을 다른 겹에서 고르고 남은 겹에 적용
sel = np.zeros(len(S), bool); outp = first.copy(); pick = []
for g in range(NF):
    tr, te = gS != g, gS == g; best = None
    for key, (x, sc) in SELS.items():
        for q in np.linspace(0, 1, 51):
            th = np.quantile(sc[tr], 1 - q) if q > 0 else np.inf; v = wacc(np.where(sc >= th, x, first), tr & mS)
            if best is None or v > best[0]: best = (v, key, q)
    _, key, q = best; x, sc = SELS[key]; th = np.quantile(sc[tr], 1 - q) if q > 0 else np.inf; s_ = sc >= th; outp[te] = np.where(s_, x, first)[te]; pick.append((key, round(float(q), 2)))
ci_, pv = boot_gain(outp)
sec5["이중겹밖(조합까지 겹밖 선택)"] = dict(이득CI=ci_, 단측p=round(pv, 4), five=five(outp), 고른것=pick)
best_key = max((kx for kx in sec5 if " | " in kx), key=lambda kx: sec5[kx]["이득CI"][0]); sec5["표본안 최선"] = [best_key, sec5[best_key]]
O["5_E11_CI"] = sec5
for kx, v in sec5.items(): print("5.", kx, v, flush=True)
json.dump(O, open(f"{OUT}/e15_summary.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DONE", flush=True)
