"""E13 분석 (PREREG.md, CPU).  학습 없는 내부 신호 (주의, logit lens, 분포) 의 T1/T2/T3/T3 같은사실 AUROC, 겹밖 고르기, H13.1/H13.2, 금 답 순위, SAT Probe 식 참고.
점수 방향은 미리 고정 ('틀림' 이 높게): 주어 주의 -, 질문 주의 -, 캔버스 주의 +, lens 일치 시작 층 +, JS +, lens 의 최종 top-1 확률 -, 엔트로피 +, top-2 차이 -, top-1 확률 -.
T1 은 e10 처럼 '가운데' 변환 -|s - m| (m = 다른 겹 1~4/5 중앙값) 만 쓴다.  T2 는 원 점수와 가운데 변환 둘 다 후보."""
import collections, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "followup_mk")
from common import grp, k, fid, NF, OUT
PIL = "--pilot" in sys.argv; D13 = f"{OUT}/e13{'_pilot' if PIL else ''}"; D7 = f"{OUT}/e7"; O = {}
R = np.load(f"{D13}/rows.npy"); assert np.load(f"{D13}/done.npy").all(), "캡처 미완료"
cfg = json.load(open(f"{D13}/config.json")); SV = cfg["steps"]; META = json.load(open(f"{D13}/meta.json"))
m7 = {}
for l in open(f"{D7}/meta.jsonl"): d = json.loads(l); m7[d["row"]] = d
corr = np.array([m7[int(i)]["new_type"] == "correct" for i in R]); kk = k[R]; g13 = grp[R]; fid13 = fid[R]
isz, isf, ism = kk == 0, kk == 5, (kk >= 1) & (kk <= 4); mw = ism & ~corr
W = np.where(isz, 44110 / isz.sum(), np.where(isf, 5165 / isf.sum(), 1.0))
sf = np.array([d["subj_found"] for d in META]); O["표본"] = dict(n=len(R), mid=int(ism.sum()), zero=int(isz.sum()), five=int(isf.sum()), mid_정답률=round(100 * corr[ism].mean(), 1), 주어찾음=round(100 * sf.mean(), 1))
print(O["표본"], flush=True)
ATT = np.load(f"{D13}/att.npy", mmap_mode="r"); LLT = np.load(f"{D13}/ll_top1.npy"); LLP = np.load(f"{D13}/ll_pfinal.npy").astype(np.float32); LLE = np.load(f"{D13}/ll_ent.npy").astype(np.float32)
LLJ = np.load(f"{D13}/ll_js.npy").astype(np.float32); LLG = np.load(f"{D13}/ll_grank.npy"); ENT = np.load(f"{D13}/ent.npy").astype(np.float32); TP = np.load(f"{D13}/top10_p.npy").astype(np.float32)
QN = ["pe", "c0", "c1", "c2", "c3"]; NL = LLT.shape[2]
# ---- 점수 ('틀림' 이 높음)
SC = {}
for si, s in enumerate(SV):
    A = np.asarray(ATT[:, si], dtype=np.float32).mean(2)                                    # [n, L, Q, S] 머리 평균
    den = np.clip(1 - A[..., 0], 1e-6, None)                                                 # sink 뺀 분모
    for qi, q in enumerate(QN + ["qmean"]):
        a = A.mean(2) if q == "qmean" else A[:, :, qi]; d_ = den.mean(2) if q == "qmean" else den[:, :, qi]
        for l_ in range(NL):
            SC[f"주어주의(-) {q} L{l_} s{s}"] = -a[:, l_, 3] / d_[:, l_]
            SC[f"질문주의(-) {q} L{l_} s{s}"] = -(a[:, l_, 2] + a[:, l_, 3]) / d_[:, l_]
            SC[f"캔버스주의 {q} L{l_} s{s}"] = a[:, l_, 4] / d_[:, l_]
        SC[f"주어주의(-) {q} 층평균 s{s}"] = -(a[:, :, 3] / d_).mean(1)
    agree = LLT[:, si] == LLT[:, si, -1:]; start = NL - np.cumprod(agree[:, ::-1], 1).sum(1)          # 끝까지 최종 top-1 과 같은 첫 층
    SC[f"lens 일치시작층 s{s}"] = start.astype(float)
    for l_ in range(4, NL - 1):
        SC[f"lens JS L{l_} s{s}"] = LLJ[:, si, l_]; SC[f"lens 최종top1확률(-) L{l_} s{s}"] = -LLP[:, si, l_]; SC[f"lens 엔트로피 L{l_} s{s}"] = LLE[:, si, l_]
    for j in range(8):
        SC[f"엔트로피 칸{j} s{s}"] = ENT[:, si, j]; SC[f"top2차이(-) 칸{j} s{s}"] = -(TP[:, si, j, 0] - TP[:, si, j, 1]); SC[f"top1확률(-) 칸{j} s{s}"] = -TP[:, si, j, 0]
    SC[f"엔트로피 칸0-3평균 s{s}"] = ENT[:, si, :4].mean(1); SC[f"top2차이(-) 칸0-3평균 s{s}"] = -(TP[:, si, :4, 0] - TP[:, si, :4, 1]).mean(1)
for nm in SC:                                                                                # 주어 못 찾은 행은 주어 주의 점수를 찾은 행 중앙값으로 (중립)
    if nm.startswith("주어주의"): SC[nm] = SC[nm].copy(); SC[nm][~sf] = np.median(SC[nm][sf])
print("점수 수", len(SC), flush=True)
names = list(SC); M = np.stack([np.nan_to_num(SC[nm], nan=0.0) for nm in names], 1).astype(np.float32)       # [n, C]
step_of = lambda nm: int(nm.split(" s")[-1])
# 가운데 변환 (겹밖 중앙값)
MC = np.zeros_like(M)
for g in range(NF):
    te = g13 == g; med = np.median(M[(g13 != g) & ism], 0); MC[te] = -np.abs(M[te] - med)
# ---- AUROC 빠른 계산
def auc_cols(y, X, w=None):
    """열마다 가중 AUROC (동점 평균).  y 0/1."""
    out = np.zeros(X.shape[1])
    for c in range(X.shape[1]): out[c] = roc_auc_score(y, X[:, c], sample_weight=w) if len(np.unique(y)) > 1 else np.nan
    return out
mi = np.where(ism)[0]
def within_cols(X, idx):
    """같은 사실 안 AUROC (틀림 양성) 의 사실 평균, 열마다.  pandas 순위로."""
    df = pd.DataFrame(X[idx]); df["f"] = fid13[idx]; y = (~corr[idx]).astype(int)
    gsz = pd.Series(y).groupby(df["f"].values).agg(["sum", "count"]); ok = gsz[(gsz["sum"] > 0) & (gsz["sum"] < gsz["count"])].index
    sel = np.isin(df["f"].values, ok); df = df[sel]; y = y[sel]; f = df.pop("f").values
    if len(f) == 0: return np.full(X.shape[1], np.nan), np.full((1, X.shape[1]), np.nan), np.array([])
    rk = df.groupby(f).rank(method="average").values                                          # [m, C]
    sr = pd.DataFrame(rk * y[:, None]).groupby(f).sum(); P = pd.Series(y).groupby(f).sum(); Nn = pd.Series(1 - y).groupby(f).sum()
    a = (sr.values - (P.values * (P.values + 1) / 2)[:, None]) / (P.values * Nn.values)[:, None]
    return a.mean(0), a, sr.index.values
# 전체 (참고용 상위표)
res = {}
res["T1"] = auc_cols(ism.astype(int), MC, W)
res["T2_raw"] = auc_cols(mw.astype(int), M, W); res["T2_mid"] = auc_cols(mw.astype(int), MC, W)
res["T3"] = auc_cols((~corr[mi]).astype(int), M[mi]); res["T3s"], _, _ = within_cols(M, mi)
print("전체 AUROC 계산 끝", flush=True)
def top(key, cols, early, k_=8):
    c = [j for j in range(len(names)) if (not early or step_of(names[j]) <= 3)]
    v = res[key]; o = sorted(c, key=lambda j: -np.nan_to_num(v[j]))[:k_]; return [(cols[j] if isinstance(cols, list) else names[j], round(float(v[j]), 3)) for j in o]
# ---- 겹밖 고르기
def nested(kind, early):
    cand = [j for j in range(len(names)) if (not early or step_of(names[j]) <= 3)]; outs = []
    for g in range(NF):
        tr = g13 != g; te = g13 == g
        if kind == "T1":
            y = ism.astype(int); v = auc_cols(y[tr], MC[tr][:, cand], W[tr]); b = cand[int(np.nanargmax(v))]; outs.append((names[b] + " (가운데)", roc_auc_score(y[te], MC[te, b], sample_weight=W[te])))
        elif kind == "T2":
            y = mw.astype(int); v1 = auc_cols(y[tr], M[tr][:, cand], W[tr]); v2 = auc_cols(y[tr], MC[tr][:, cand], W[tr])
            if np.nanmax(v1) >= np.nanmax(v2): b = cand[int(np.nanargmax(v1))]; outs.append((names[b], roc_auc_score(y[te], M[te, b], sample_weight=W[te])))
            else: b = cand[int(np.nanargmax(v2))]; outs.append((names[b] + " (가운데)", roc_auc_score(y[te], MC[te, b], sample_weight=W[te])))
        elif kind == "T3":
            itr = np.intersect1d(mi, np.where(tr)[0]); ite = np.intersect1d(mi, np.where(te)[0]); v = auc_cols((~corr[itr]).astype(int), M[itr][:, cand]); b = cand[int(np.nanargmax(v))]
            outs.append((names[b], roc_auc_score((~corr[ite]).astype(int), M[ite, b])))
        else:
            itr = np.intersect1d(mi, np.where(tr)[0]); ite = np.intersect1d(mi, np.where(te)[0]); v, _, _ = within_cols(M[:, cand], itr)
            if np.all(np.isnan(v)): outs.append(("없음", np.nan)); continue
            b = cand[int(np.nanargmax(v))]
            w_, _, _ = within_cols(M[:, [b]], ite); outs.append((names[b], float(w_[0])))
    return dict(평균=round(float(np.nanmean([a for _, a in outs])), 3) if not all(np.isnan(a) for _, a in outs) else None, 겹별=[round(float(a), 3) for _, a in outs], 고른점수=[b for b, _ in outs])
O["감지"] = {}
for kind, key in (("T1", "T1"), ("T2", "T2_raw"), ("T3", "T3"), ("T3s", "T3s")):
    for early in (True, False):
        tag = f"{kind} {'step<=3' if early else 'step<=8'}"
        O["감지"][tag] = dict(상위=top(key, names, early), 겹밖최선=nested(kind, early))
        if kind == "T2": O["감지"][tag]["상위_가운데"] = top("T2_mid", names, early)
        print(tag, O["감지"][tag]["겹밖최선"], O["감지"][tag]["상위"][:3], flush=True)
O["판정"] = {"H13_1": bool((O["감지"]["T3s step<=3"]["겹밖최선"]["평균"] or 0) >= 0.70)}
# ---- H13.2: 미리 정한 하나의 점수 (step 0, 머리·층·질의 평균 주어 주의, sink 뺀 비율).  같은 사실 안 AUROC, 사실 부트스트랩.
h2 = names.index("주어주의(-) qmean 층평균 s0"); w_, per, _ = within_cols(M[:, [h2]], mi); per = per[:, 0]
bs = [per[np.random.default_rng(b).integers(0, len(per), len(per))].mean() for b in range(2000)]
O["H13_2"] = dict(점수="주어주의(-) qmean 층평균 s0", 같은사실AUROC=round(float(per.mean()), 3), CI=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], 사실수=len(per),
                  주어찾음_행만=None)
okf = sf[mi]; w2, per2, _ = within_cols(M[:, [h2]], mi[okf]); O["H13_2"]["주어찾음_행만"] = round(float(per2[:, 0].mean()), 3)
O["판정"]["H13_2"] = bool(per.mean() > 0.5 and np.percentile(bs, 2.5) > 0.5)
# 층별 주어 주의 (step 0, qmean) 같은 사실 AUROC 곡선 (서술)
cl = [names.index(f"주어주의(-) qmean L{l_} s0") for l_ in range(NL)]; wv, _, _ = within_cols(M[:, cl], mi)
O["주어주의_층별_같은사실_s0"] = [round(float(x), 3) for x in wv]
# 그룹 평균 주어 주의 (s0, qmean, 층평균) 서술
O["주어주의_평균_s0"] = {nm: round(float(-M[ix, h2].mean()), 4) for nm, ix in (("0/5", isz), ("1~4/5 정답", ism & corr), ("1~4/5 오답", mw), ("5/5", isf))}
print("H13.2", O["H13_2"], flush=True)
# ---- 금 답 첫 토큰 순위 (step 0, 칸 0).  0 = top-1.
gr = LLG[:, 0, -1]; gbest = LLG[:, 0, 14:].min(1)
desc = {}
for nm, ix in (("0/5", isz), ("1~4/5 정답", ism & corr), ("1~4/5 오답", mw), ("5/5", isf)):
    desc[nm] = {f"top{t}": round(100 * float((gr[ix] < t).mean()), 1) for t in (1, 2, 5, 10)}
    desc[nm]["중간층(14~26)에선 top1, 마지막층은 아님"] = round(100 * float(((gbest[ix] == 0) & (gr[ix] > 0)).mean()), 1); desc[nm]["순위 중앙값"] = float(np.median(gr[ix]))
O["금답순위_s0_칸0"] = desc; print("금답순위", desc, flush=True)
# step 별 1~4/5 오답의 top-2 안 비율
O["1~4/5오답_금답top2_step별"] = {f"s{s}": round(100 * float((LLG[mw, si, -1] < 2).mean()), 1) for si, s in enumerate(SV)}
# ---- 참고: SAT Probe 식 (머리별 주어 주의, 로지스틱, 주어 겹밖, 학습 있음)
ref = {}
for s in (0, 3):
    si = SV.index(s); A = np.asarray(ATT[:, si], dtype=np.float32); den = np.clip(1 - A[..., 0], 1e-6, None)
    Xf = np.concatenate([(A[:, :, :, q, 3] / den[:, :, :, q]).reshape(len(R), -1) for q in (0, 1)], 1)       # pe, c0 질의, [n, 2*L*H]
    Xf[~sf] = Xf[sf].mean(0)
    sc = np.zeros(len(R))
    for g in range(NF):
        tr = np.intersect1d(mi, np.where(g13 != g)[0]); te = np.where(g13 == g)[0]; mu = Xf[tr].mean(0); sd = Xf[tr].std(0) + 1e-6
        clf = LogisticRegression(C=0.01, max_iter=2000).fit((Xf[tr] - mu) / sd, (~corr[tr]).astype(int)); sc[te] = clf.decision_function((Xf[te] - mu) / sd)
    w_, _, _ = within_cols(sc[:, None], mi)
    ref[f"SAT식 s{s}"] = dict(T3=round(float(roc_auc_score((~corr[mi]).astype(int), sc[mi])), 3), T3_같은사실=round(float(w_[0]), 3))
O["참고_SAT식_probe"] = ref; print("SAT식", ref, flush=True)
json.dump(O, open(f"{OUT}/e13_summary{'_pilot' if PIL else ''}.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("DONE", flush=True)
